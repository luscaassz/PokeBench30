from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from pokebench30 import __version__
from pokebench30.representation.schema import build_feature_schema
from pokebench30.rl.model_io import ModelMetadata, load_model_metadata, validate_model_metadata
from pokebench30.rl.policy_specs import (
    DEFAULT_POLICY_ARCHITECTURES,
    get_policy_spec,
    validate_architectures,
)
from pokebench30.rl.reward import RewardConfig
from pokebench30.teams.fixed_pool import DEFAULT_POOL_ROOT, load_pool_manifest, verify_pool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "rl_baseline.yaml"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "policy_results"


@dataclass(frozen=True)
class Stage:
    opponent: str
    timesteps: int


@dataclass(frozen=True)
class EvalRow:
    architecture: str
    seed: int
    checkpoint: int
    opponent: str
    evaluation_kind: str
    repeats_per_leg: int
    battles: int
    mirror_cycles: int
    wins_rl: int
    wins_opponent: int
    draws: int
    raw_win_rate_rl: float
    mirror_score_rl: float
    team_assignment_gap: float
    source: str
    model_dir: str
    evaluation_dir: str


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _monitor_path(prefix: Path) -> Path | None:
    candidates = [
        prefix,
        Path(str(prefix) + ".monitor.csv"),
        prefix.with_suffix(prefix.suffix + ".monitor.csv") if prefix.suffix else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def _read_monitor(prefix: Path) -> list[dict[str, str]]:
    path = _monitor_path(prefix)
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        if not first.startswith("#"):
            handle.seek(0)
        return list(csv.DictReader(handle))


def _training_stats(rows: list[dict[str, str]], last_window: int = 20) -> dict[str, Any]:
    rewards = [float(row["r"]) for row in rows if row.get("r") not in (None, "")]
    lengths = [float(row["l"]) for row in rows if row.get("l") not in (None, "")]
    return {
        "episodes": len(rows),
        "mean_reward": statistics.fmean(rewards) if rewards else "",
        "last_window_reward": statistics.fmean(rewards[-last_window:]) if rewards else "",
        "mean_episode_length": statistics.fmean(lengths) if lengths else "",
    }


def parse_fixed_curriculum(config: dict[str, Any]) -> list[Stage]:
    block = config["policy_study"]["curriculum"]
    return [
        Stage(opponent=str(stage["opponent"]), timesteps=int(stage["timesteps"]))
        for stage in block
    ]


def validate_curriculum(
    stages: list[Stage],
    *,
    total_timesteps: int,
    checkpoint_interval: int,
) -> None:
    if not stages:
        raise ValueError("Policy study curriculum cannot be empty.")
    if sum(stage.timesteps for stage in stages) != total_timesteps:
        raise ValueError(
            "Policy-study curriculum total does not match total_timesteps."
        )
    cumulative = 0
    for stage in stages:
        if stage.timesteps <= 0:
            raise ValueError("Curriculum stage timesteps must be > 0.")
        cumulative += stage.timesteps
        if cumulative != total_timesteps and cumulative % checkpoint_interval != 0:
            raise ValueError(
                f"Curriculum transition at {cumulative} is not aligned with "
                f"checkpoint_interval={checkpoint_interval}."
            )


def opponent_at(stages: list[Stage], nominal_step: int) -> str:
    if nominal_step < 0:
        raise ValueError("nominal_step must be >= 0.")
    cumulative = 0
    for stage in stages:
        end = cumulative + stage.timesteps
        if nominal_step < end:
            return stage.opponent
        cumulative = end
    raise ValueError(
        f"nominal_step={nominal_step} outside curriculum total={cumulative}."
    )


def checkpoints(total_timesteps: int, interval: int) -> list[int]:
    if total_timesteps <= 0 or interval <= 0:
        raise ValueError("Timesteps and interval must be > 0.")
    if total_timesteps % interval != 0:
        raise ValueError("total_timesteps must be divisible by checkpoint_interval.")
    return list(range(interval, total_timesteps + 1, interval))


def _make_env(
    *,
    battle_format: str,
    opponent_key: str,
    representation_version: str,
    architecture: str,
    reward: RewardConfig,
    log_level: int,
    monitor_prefix: Path,
):
    try:
        from poke_env.environment import SingleAgentWrapper
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "RL dependencies are not installed. Run:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    from pokebench30.rl.env import UniversalRLSinglesEnv
    from pokebench30.rl.mask_adapter import SB3MaskAdapter
    from pokebench30.rl.temporal_policy import wrap_env_for_policy
    from pokebench30.rl.training import make_opponent

    base_env = UniversalRLSinglesEnv(
        battle_format=battle_format,
        reward_config=reward,
        representation_version=representation_version,
        log_level=log_level,
        open_timeout=None,
    )
    opponent = make_opponent(opponent_key, battle_format=battle_format)
    single_env = SingleAgentWrapper(base_env, opponent)
    monitored = Monitor(single_env, filename=str(monitor_prefix))
    masked = SB3MaskAdapter(monitored)
    return wrap_env_for_policy(masked, get_policy_spec(architecture))


def _new_model(
    *,
    env,
    architecture: str,
    base_dim: int,
    seed: int,
    args: argparse.Namespace,
    algo: dict[str, Any],
    tensorboard_dir: Path,
):
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "sb3-contrib is required. Run: python -m pip install -r requirements.txt"
        ) from exc

    from pokebench30.rl.temporal_policy import policy_kwargs_for_spec

    spec = get_policy_spec(architecture)
    policy_kwargs = policy_kwargs_for_spec(spec, base_dim=base_dim)

    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=float(args.learning_rate),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(algo.get("n_epochs", 10)),
        gamma=float(algo.get("gamma", 0.99)),
        gae_lambda=float(algo.get("gae_lambda", 0.95)),
        clip_range=float(algo.get("clip_range", 0.2)),
        ent_coef=float(algo.get("ent_coef", 0.01)),
        vf_coef=float(algo.get("vf_coef", 0.5)),
        max_grad_norm=float(algo.get("max_grad_norm", 0.5)),
        policy_kwargs=policy_kwargs,
        seed=int(seed),
        device=args.device,
        tensorboard_log=str(tensorboard_dir),
        verbose=1,
    )


def _parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.policy.parameters()))


def _checkpoint_metadata(
    *,
    generation: int,
    representation_version: str,
    architecture: str,
    nominal_checkpoint: int,
    actual_num_timesteps: int,
    seed: int,
    reward: RewardConfig,
    stages: list[Stage],
    args: argparse.Namespace,
    parameter_count: int,
) -> tuple[ModelMetadata, dict[str, Any]]:
    schema = build_feature_schema(representation_version)
    spec = get_policy_spec(architecture)
    battle_format = f"gen{generation}randombattle"

    policy_config = spec.to_dict()
    policy_config["parameter_count"] = int(parameter_count)

    metadata = ModelMetadata(
        project_version=__version__,
        algorithm="MaskablePPO",
        training_generation=generation,
        training_battle_format=battle_format,
        training_opponent="curriculum:random_to_type_aware",
        total_timesteps=int(nominal_checkpoint),
        seed=int(seed),
        observation_schema_version=schema.version,
        observation_schema_sha256=schema.sha256,
        observation_dim=schema.observation_dim,
        action_dim=schema.action_dim,
        reward=reward.to_dict(),
        library_versions={
            "poke-env": _version("poke-env"),
            "gymnasium": _version("gymnasium"),
            "stable-baselines3": _version("stable-baselines3"),
            "sb3-contrib": _version("sb3-contrib"),
            "torch": _version("torch"),
            "numpy": _version("numpy"),
        },
        notes=[
            "PokéBench30 v0.5.5 policy-architecture checkpoint.",
            f"Architecture: {architecture}.",
            "UBR 1.2, reward, curriculum and PPO hyperparameters are held fixed.",
            "Invalid actions are excluded by MaskablePPO action masking.",
            "Temporal architectures use a fixed four-decision window.",
            "history_lstm is a fixed-window LSTM encoder, not stateful RecurrentPPO.",
            "Nominal checkpoints are labels; PPO collects full n_steps rollouts.",
        ],
        policy_architecture=architecture,
        policy_config=policy_config,
        model_input_dim=spec.model_input_dim(schema.observation_dim),
    )

    architecture_meta = {
        "project": "PokeBench30",
        "version": __version__,
        "experiment_type": "policy_architecture_study",
        "architecture": architecture,
        "architecture_spec": spec.to_dict(),
        "policy_parameter_count": int(parameter_count),
        "curriculum": [asdict(stage) for stage in stages],
        "nominal_checkpoint_timesteps": int(nominal_checkpoint),
        "actual_model_num_timesteps": int(actual_num_timesteps),
        "seed": int(seed),
        "representation_version": representation_version,
        "ubr_observation_dim": schema.observation_dim,
        "model_input_dim": spec.model_input_dim(schema.observation_dim),
        "learning_rate": float(args.learning_rate),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
    }
    return metadata, architecture_meta


def _save_checkpoint(
    *,
    model,
    checkpoint_dir: Path,
    generation: int,
    representation_version: str,
    architecture: str,
    nominal_checkpoint: int,
    seed: int,
    reward: RewardConfig,
    stages: list[Stage],
    args: argparse.Namespace,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_dir / "maskable_ppo"))
    parameters = _parameter_count(model)
    metadata, architecture_meta = _checkpoint_metadata(
        generation=generation,
        representation_version=representation_version,
        architecture=architecture,
        nominal_checkpoint=nominal_checkpoint,
        actual_num_timesteps=int(model.num_timesteps),
        seed=seed,
        reward=reward,
        stages=stages,
        args=args,
        parameter_count=parameters,
    )
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (checkpoint_dir / "architecture_metadata.json").write_text(
        json.dumps(architecture_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_eval(
    *,
    architecture: str,
    seed: int,
    checkpoint: int,
    opponent: str,
    kind: str,
    repeats: int,
    model_dir: Path,
    eval_dir: Path,
    source: str = "trained_v0.5.5",
) -> EvalRow:
    rows = _read_csv(eval_dir / "controlled_summary.csv")
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one controlled_summary row at {eval_dir}, got {len(rows)}."
        )
    row = rows[0]
    return EvalRow(
        architecture=architecture,
        seed=int(seed),
        checkpoint=int(checkpoint),
        opponent=opponent,
        evaluation_kind=kind,
        repeats_per_leg=int(repeats),
        battles=int(row["battles"]),
        mirror_cycles=int(row["mirror_cycles"]),
        wins_rl=int(row["wins_agent_1"]),
        wins_opponent=int(row["wins_agent_2"]),
        draws=int(row["draws"]),
        raw_win_rate_rl=float(row["raw_win_rate_agent_1"]),
        mirror_score_rl=float(row["normalized_mirror_score_agent_1"]),
        team_assignment_gap=float(row["agent_1_team_assignment_gap"]),
        source=source,
        model_dir=str(model_dir),
        evaluation_dir=str(eval_dir),
    )


def _evaluate_checkpoint(
    *,
    model_dir: Path,
    eval_root: Path,
    architecture: str,
    seed: int,
    checkpoint: int,
    opponent: str,
    kind: str,
    repeats: int,
    args: argparse.Namespace,
) -> EvalRow:
    eval_name = f"step_{checkpoint:06d}_vs_{opponent}"
    eval_dir = eval_root / eval_name
    summary = eval_dir / "controlled_summary.csv"

    if not summary.exists():
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "run_rl_controlled.py"),
            "--model", str(model_dir / "maskable_ppo.zip"),
            "--generation", str(args.generation),
            "--opponent", opponent,
            "--repeats", str(repeats),
            "--concurrency", str(args.concurrency),
            "--pool-root", str(Path(args.pool_root).resolve()),
            "--output-root", str(eval_root),
            "--run-name", eval_name,
            "--no-action-log",
            "--no-plots",
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    else:
        print(f"[RESUME] Existing evaluation accepted: {eval_dir}")

    return _load_eval(
        architecture=architecture,
        seed=seed,
        checkpoint=checkpoint,
        opponent=opponent,
        kind=kind,
        repeats=repeats,
        model_dir=model_dir,
        eval_dir=eval_dir,
    )


def _validate_v054_baseline(
    source_root: Path,
    *,
    args: argparse.Namespace,
    stages: list[Stage],
    pool_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = source_root / "study_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"v0.5.4 study_manifest.json not found: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    def expect(label: str, observed: Any, expected: Any) -> None:
        if observed != expected:
            errors.append(f"{label}: observed={observed!r}, expected={expected!r}")

    expect("generation", int(manifest.get("generation", -1)), int(args.generation))
    expect("representation", manifest.get("representation"), args.ubr)
    expect("seeds", manifest.get("seeds"), [int(x) for x in args.seeds])
    expect("total_timesteps", int(manifest.get("total_timesteps", -1)), int(args.total_timesteps))
    expect(
        "checkpoint_interval",
        int(manifest.get("checkpoint_interval", -1)),
        int(args.checkpoint_interval),
    )
    expect(
        "checkpoint_repeats_per_leg",
        int(manifest.get("checkpoint_repeats_per_leg", -1)),
        int(args.checkpoint_repeats),
    )
    expect(
        "final_repeats_per_leg",
        int(manifest.get("final_repeats_per_leg", -1)),
        int(args.final_repeats),
    )
    expect("evaluation_opponents", manifest.get("evaluation_opponents"), list(args.eval_opponents))

    expected_stages = [asdict(stage) for stage in stages]
    observed_stages = (
        manifest.get("regimes", {})
        .get("random_to_type_aware")
    )
    expect("random_to_type_aware curriculum", observed_stages, expected_stages)

    ppo = manifest.get("ppo", {})
    expect("n_steps", int(ppo.get("n_steps", -1)), int(args.n_steps))
    expect("batch_size", int(ppo.get("batch_size", -1)), int(args.batch_size))
    observed_lr = float(ppo.get("learning_rate", float("nan")))
    if abs(observed_lr - float(args.learning_rate)) > 1e-12:
        errors.append(
            f"learning_rate: observed={observed_lr}, expected={args.learning_rate}"
        )

    pool = manifest.get("fixed_team_pool", {})
    expect(
        "fixed_team_pool.config_sha256",
        pool.get("config_sha256"),
        pool_manifest.get("config_sha256"),
    )
    expect(
        "fixed_team_pool.showdown_commit",
        pool.get("showdown_commit"),
        pool_manifest.get("showdown_commit"),
    )

    if errors:
        raise ValueError(
            "The supplied v0.5.4 experiment cannot be imported as the "
            "v0.5.5 MLP baseline:\n- " + "\n- ".join(errors)
        )
    return manifest


def import_v054_baseline(
    source_root: Path,
    *,
    args: argparse.Namespace,
    stages: list[Stage],
    pool_manifest: dict[str, Any],
    snapshot_dir: Path | None = None,
) -> tuple[list[EvalRow], list[dict[str, Any]]]:
    source_root = Path(source_root).resolve()
    manifest = _validate_v054_baseline(
        source_root,
        args=args,
        stages=stages,
        pool_manifest=pool_manifest,
    )
    aggregate = source_root / "aggregate"
    learning_path = aggregate / "learning_curve.csv"
    training_path = aggregate / "training_progress.csv"
    if not learning_path.exists() or not training_path.exists():
        raise FileNotFoundError(
            "v0.5.4 baseline must contain aggregate/learning_curve.csv and "
            "aggregate/training_progress.csv."
        )

    eval_rows: list[EvalRow] = []
    for row in _read_csv(learning_path):
        if row["regime"] != "random_to_type_aware":
            continue
        eval_rows.append(
            EvalRow(
                architecture="mlp_baseline",
                seed=int(row["seed"]),
                checkpoint=int(row["checkpoint"]),
                opponent=row["opponent"],
                evaluation_kind=row["evaluation_kind"],
                repeats_per_leg=int(row["repeats_per_leg"]),
                battles=int(row["battles"]),
                mirror_cycles=int(row["mirror_cycles"]),
                wins_rl=int(row["wins_rl"]),
                wins_opponent=int(row["wins_opponent"]),
                draws=int(row["draws"]),
                raw_win_rate_rl=float(row["raw_win_rate_rl"]),
                mirror_score_rl=float(row["mirror_score_rl"]),
                team_assignment_gap=float(row["team_assignment_gap"]),
                source="imported_v0.5.4",
                model_dir=row["model_dir"],
                evaluation_dir=row["evaluation_dir"],
            )
        )

    training_rows: list[dict[str, Any]] = []
    for row in _read_csv(training_path):
        if row["regime"] != "random_to_type_aware":
            continue
        training_rows.append({
            "architecture": "mlp_baseline",
            "seed": int(row["seed"]),
            "checkpoint": int(row["checkpoint"]),
            "training_opponent": row["training_opponent"],
            "requested_segment_timesteps": int(row["requested_segment_timesteps"]),
            "actual_model_num_timesteps": row["actual_model_num_timesteps"],
            "episodes_segment": int(row["episodes_segment"]) if row["episodes_segment"] else "",
            "mean_reward_segment": float(row["mean_reward_segment"]) if row["mean_reward_segment"] else "",
            "last_window_reward_segment": float(row["last_window_reward_segment"]) if row["last_window_reward_segment"] else "",
            "mean_episode_length_segment": float(row["mean_episode_length_segment"]) if row["mean_episode_length_segment"] else "",
            "source": "imported_v0.5.4",
            "monitor_file": row["monitor_file"],
        })

    expected_eval = (
        len(args.seeds)
        * len(checkpoints(args.total_timesteps, args.checkpoint_interval))
        * len(args.eval_opponents)
    )
    expected_training = (
        len(args.seeds)
        * len(checkpoints(args.total_timesteps, args.checkpoint_interval))
    )
    if len(eval_rows) != expected_eval:
        raise RuntimeError(
            f"Expected {expected_eval} imported evaluation rows, got {len(eval_rows)}."
        )
    if len(training_rows) != expected_training:
        raise RuntimeError(
            f"Expected {expected_training} imported training rows, got {len(training_rows)}."
        )

    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / "study_manifest.json", snapshot_dir / "source_study_manifest.json")
        filtered_eval = [asdict(row) for row in eval_rows]
        _write_csv(
            snapshot_dir / "learning_curve_mlp_baseline.csv",
            filtered_eval,
            list(asdict(eval_rows[0]).keys()),
        )
        _write_csv(
            snapshot_dir / "training_progress_mlp_baseline.csv",
            training_rows,
            list(training_rows[0].keys()),
        )
        (snapshot_dir / "import_info.json").write_text(
            json.dumps(
                {
                    "source_experiment_dir": str(source_root),
                    "source_regime": "random_to_type_aware",
                    "mapped_architecture": "mlp_baseline",
                    "source_manifest": manifest,
                    "note": (
                        "Aggregate rows are snapshotted. Original v0.5.4 model "
                        "directories remain at their source paths."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return eval_rows, training_rows


def _write_aggregates(
    root: Path,
    eval_rows: list[EvalRow],
    training_rows: list[dict[str, Any]],
    baseline_architecture: str = "mlp_baseline",
) -> None:
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)

    eval_dicts = [asdict(row) for row in eval_rows]
    eval_fields = list(asdict(eval_rows[0]).keys()) if eval_rows else [
        "architecture", "seed", "checkpoint", "opponent", "evaluation_kind",
        "repeats_per_leg", "battles", "mirror_cycles", "wins_rl",
        "wins_opponent", "draws", "raw_win_rate_rl", "mirror_score_rl",
        "team_assignment_gap", "source", "model_dir", "evaluation_dir",
    ]
    _write_csv(aggregate / "learning_curve.csv", eval_dicts, eval_fields)

    training_fields = [
        "architecture", "seed", "checkpoint", "training_opponent",
        "requested_segment_timesteps", "actual_model_num_timesteps",
        "episodes_segment", "mean_reward_segment", "last_window_reward_segment",
        "mean_episode_length_segment", "source", "monitor_file",
    ]
    _write_csv(aggregate / "training_progress.csv", training_rows, training_fields)

    grouped: dict[tuple[str, int, str, str], list[EvalRow]] = {}
    for row in eval_rows:
        grouped.setdefault(
            (row.architecture, row.checkpoint, row.opponent, row.evaluation_kind),
            [],
        ).append(row)

    agg_rows: list[dict[str, Any]] = []
    for (architecture, checkpoint, opponent, kind), rows in sorted(grouped.items()):
        scores = [row.mirror_score_rl for row in rows]
        agg_rows.append({
            "architecture": architecture,
            "checkpoint": checkpoint,
            "opponent": opponent,
            "evaluation_kind": kind,
            "n_seeds": len(rows),
            "mean_mirror_score": statistics.fmean(scores),
            "sample_sd_mirror_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min_mirror_score": min(scores),
            "max_mirror_score": max(scores),
            "total_wins_rl": sum(row.wins_rl for row in rows),
            "total_wins_opponent": sum(row.wins_opponent for row in rows),
            "total_draws": sum(row.draws for row in rows),
            "total_battles": sum(row.battles for row in rows),
        })

    _write_csv(
        aggregate / "aggregate_learning_curve.csv",
        agg_rows,
        [
            "architecture", "checkpoint", "opponent", "evaluation_kind",
            "n_seeds", "mean_mirror_score", "sample_sd_mirror_score",
            "min_mirror_score", "max_mirror_score", "total_wins_rl",
            "total_wins_opponent", "total_draws", "total_battles",
        ],
    )

    final_rows = [row for row in eval_rows if row.evaluation_kind == "final"]
    final_grouped: dict[tuple[str, str], list[EvalRow]] = {}
    for row in final_rows:
        final_grouped.setdefault((row.architecture, row.opponent), []).append(row)

    final_summary: list[dict[str, Any]] = []
    for (architecture, opponent), rows in sorted(final_grouped.items()):
        scores = [row.mirror_score_rl for row in rows]
        final_summary.append({
            "architecture": architecture,
            "opponent": opponent,
            "n_seeds": len(rows),
            "mean_mirror_score": statistics.fmean(scores),
            "sample_sd_mirror_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min_mirror_score": min(scores),
            "max_mirror_score": max(scores),
            "total_wins_rl": sum(row.wins_rl for row in rows),
            "total_wins_opponent": sum(row.wins_opponent for row in rows),
            "total_draws": sum(row.draws for row in rows),
            "total_battles": sum(row.battles for row in rows),
        })

    _write_csv(
        aggregate / "final_summary.csv",
        final_summary,
        [
            "architecture", "opponent", "n_seeds", "mean_mirror_score",
            "sample_sd_mirror_score", "min_mirror_score", "max_mirror_score",
            "total_wins_rl", "total_wins_opponent", "total_draws",
            "total_battles",
        ],
    )

    by_key = {
        (row.architecture, row.seed, row.opponent): row
        for row in final_rows
    }
    architectures = sorted({row.architecture for row in final_rows})
    seeds = sorted({row.seed for row in final_rows})
    opponents = sorted({row.opponent for row in final_rows})

    paired_rows: list[dict[str, Any]] = []
    for architecture in architectures:
        if architecture == baseline_architecture:
            continue
        for opponent in opponents:
            for seed in seeds:
                baseline = by_key.get((baseline_architecture, seed, opponent))
                candidate = by_key.get((architecture, seed, opponent))
                if baseline is None or candidate is None:
                    continue
                paired_rows.append({
                    "architecture": architecture,
                    "seed": seed,
                    "opponent": opponent,
                    "baseline_score": baseline.mirror_score_rl,
                    "architecture_score": candidate.mirror_score_rl,
                    "difference_vs_baseline": (
                        candidate.mirror_score_rl - baseline.mirror_score_rl
                    ),
                })

    _write_csv(
        aggregate / "paired_architecture_differences.csv",
        paired_rows,
        [
            "architecture", "seed", "opponent", "baseline_score",
            "architecture_score", "difference_vs_baseline",
        ],
    )

    comparison_rows: list[dict[str, Any]] = []
    for architecture in architectures:
        if architecture == baseline_architecture:
            continue
        for opponent in opponents:
            rows = [
                row for row in paired_rows
                if row["architecture"] == architecture
                and row["opponent"] == opponent
            ]
            if not rows:
                continue
            diffs = [float(row["difference_vs_baseline"]) for row in rows]
            comparison_rows.append({
                "architecture": architecture,
                "opponent": opponent,
                "paired_seeds": len(rows),
                "mean_baseline": statistics.fmean(
                    float(row["baseline_score"]) for row in rows
                ),
                "mean_architecture": statistics.fmean(
                    float(row["architecture_score"]) for row in rows
                ),
                "mean_difference_vs_baseline": statistics.fmean(diffs),
                "sample_sd_difference": (
                    statistics.stdev(diffs) if len(diffs) > 1 else 0.0
                ),
            })

    _write_csv(
        aggregate / "architecture_comparison.csv",
        comparison_rows,
        [
            "architecture", "opponent", "paired_seeds", "mean_baseline",
            "mean_architecture", "mean_difference_vs_baseline",
            "sample_sd_difference",
        ],
    )

    figures = aggregate / "figures"
    figures.mkdir(exist_ok=True)

    for opponent in sorted({row.opponent for row in eval_rows}):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for architecture in sorted({row.architecture for row in eval_rows}):
            points = [
                row for row in agg_rows
                if row["architecture"] == architecture
                and row["opponent"] == opponent
            ]
            points.sort(key=lambda row: int(row["checkpoint"]))
            if not points:
                continue
            ax.errorbar(
                [int(row["checkpoint"]) for row in points],
                [float(row["mean_mirror_score"]) for row in points],
                yerr=[float(row["sample_sd_mirror_score"]) for row in points],
                marker="o",
                capsize=3,
                label=architecture,
            )
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Nominal training timesteps")
        ax.set_ylabel("Mean mirrored score (± seed SD)")
        ax.set_title(f"Policy architecture learning curves vs {opponent}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"learning_curve_vs_{opponent}.png", dpi=160)
        plt.close(fig)


def _study_manifest(
    args: argparse.Namespace,
    stages: list[Stage],
    pool: dict[str, Any],
) -> dict[str, Any]:
    return {
        "project": "PokeBench30",
        "version": __version__,
        "experiment_type": "policy_architecture_study",
        "generation": int(args.generation),
        "representation": args.ubr,
        "reward_profile": "v0.5_dense_reward_unchanged",
        "architectures": list(args.architectures),
        "architecture_specs": {
            name: get_policy_spec(name).to_dict()
            for name in args.architectures
        },
        "seeds": [int(x) for x in args.seeds],
        "total_timesteps": int(args.total_timesteps),
        "checkpoint_interval": int(args.checkpoint_interval),
        "checkpoint_repeats_per_leg": int(args.checkpoint_repeats),
        "final_repeats_per_leg": int(args.final_repeats),
        "evaluation_opponents": list(args.eval_opponents),
        "curriculum": [asdict(stage) for stage in stages],
        "ppo": {
            "learning_rate": float(args.learning_rate),
            "learning_rate_schedule": "constant",
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
        },
        "baseline_curriculum_dir": (
            str(Path(args.baseline_curriculum_dir).resolve())
            if args.baseline_curriculum_dir
            else None
        ),
        "fixed_team_pool": {
            "config_sha256": pool.get("config_sha256"),
            "showdown_commit": pool.get("showdown_commit"),
            "showdown_dirty": pool.get("showdown_dirty"),
        },
        "design_notes": [
            "v0.5.5 isolates policy architecture while keeping UBR, reward, curriculum and PPO hyperparameters fixed.",
            "Curriculum is Random 100k -> TypeAware 200k.",
            "mlp_baseline is the v0.5-v0.5.4 policy.",
            "mlp_large tests feed-forward capacity.",
            "history_mlp tests short temporal context without sequence-specific inductive bias.",
            "history_lstm uses an LSTM over the same fixed four-decision history window.",
            "A fully stateful Maskable RecurrentPPO is intentionally not used because sb3-contrib 2.9 does not natively support recurrent MaskablePPO.",
            "Intermediate checkpoint evaluations are diagnostic; final evaluations use the full mirrored protocol.",
        ],
    }


def _same_manifest(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = [
        "generation", "representation", "architectures", "architecture_specs",
        "seeds", "total_timesteps", "checkpoint_interval",
        "checkpoint_repeats_per_leg", "final_repeats_per_leg",
        "evaluation_opponents", "curriculum", "ppo",
        "baseline_curriculum_dir", "fixed_team_pool",
    ]

    def canonical(value: Any) -> Any:
        return json.loads(
            json.dumps(value, sort_keys=True, ensure_ascii=False)
        )

    return all(
        canonical(a.get(key)) == canonical(b.get(key))
        for key in keys
    )


def run_study(args: argparse.Namespace) -> Path:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "RL dependencies are not installed. Run:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    # Import custom extractor classes before any model deserialization.
    from pokebench30.rl import temporal_policy as _temporal_policy  # noqa: F401

    config = load_config(Path(args.config))
    algo = config["algorithm"]
    reward = RewardConfig.from_mapping(config.get("reward"))
    schema = build_feature_schema(args.ubr)
    architectures = validate_architectures(args.architectures)
    args.architectures = architectures

    stages = parse_fixed_curriculum(config)
    if args.smoke:
        args.seeds = [int(args.seeds[0])]
        args.total_timesteps = 6144
        args.checkpoint_interval = 2048
        args.checkpoint_repeats = 1
        args.final_repeats = 1
        args.n_steps = 256
        args.batch_size = 64
        stages = [
            Stage("random", 2048),
            Stage("type_aware", 4096),
        ]
        # A smoke is self-contained; do not import a 300k baseline.
        args.baseline_curriculum_dir = None

    validate_curriculum(
        stages,
        total_timesteps=args.total_timesteps,
        checkpoint_interval=args.checkpoint_interval,
    )

    pool_root = Path(args.pool_root).resolve()
    problems = verify_pool(pool_root)
    if problems:
        raise RuntimeError(
            "Fixed-team pool failed integrity check:\n- "
            + "\n- ".join(problems)
        )
    pool_manifest = load_pool_manifest(pool_root)

    study_root = (
        Path(args.study_dir).resolve()
        if args.study_dir
        else Path(args.output_root).resolve()
        / f"policy_gen{args.generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    study_root.mkdir(parents=True, exist_ok=True)

    requested_manifest = _study_manifest(args, stages, pool_manifest)
    manifest_path = study_root / "study_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _same_manifest(existing, requested_manifest):
            raise ValueError(
                "The selected --study-dir contains a different policy-study "
                "configuration. Use a new directory."
            )
    else:
        manifest_path.write_text(
            json.dumps(requested_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    all_eval_rows: list[EvalRow] = []
    all_training_rows: list[dict[str, Any]] = []
    imported_architecture: str | None = None

    if args.baseline_curriculum_dir and "mlp_baseline" in architectures:
        imported_eval, imported_training = import_v054_baseline(
            Path(args.baseline_curriculum_dir),
            args=args,
            stages=stages,
            pool_manifest=pool_manifest,
            snapshot_dir=study_root / "imported_mlp_baseline",
        )
        all_eval_rows.extend(imported_eval)
        all_training_rows.extend(imported_training)
        imported_architecture = "mlp_baseline"
        print(
            f"[IMPORT] v0.5.4 random_to_type_aware -> mlp_baseline: "
            f"{len(imported_training)} training rows, "
            f"{len(imported_eval)} evaluation rows."
        )

    nominal_checkpoints = checkpoints(
        args.total_timesteps,
        args.checkpoint_interval,
    )
    battle_format = f"gen{args.generation}randombattle"

    train_architectures = [
        architecture
        for architecture in architectures
        if architecture != imported_architecture
    ]

    print(f"[INFO] Policy-study directory: {study_root}")
    print(f"[INFO] Architectures: {architectures}")
    print(f"[INFO] Architectures to train now: {train_architectures}")
    print(f"[INFO] Seeds: {args.seeds}")
    print(f"[INFO] Curriculum: {[asdict(stage) for stage in stages]}")

    for architecture in train_architectures:
        spec = get_policy_spec(architecture)

        for seed in args.seeds:
            run_root = study_root / "runs" / architecture / f"seed_{int(seed)}"
            checkpoints_root = run_root / "checkpoints"
            logs_root = run_root / "training_logs"
            tensorboard_root = run_root / "tensorboard"
            eval_root = run_root / "evaluations"
            for folder in (
                checkpoints_root,
                logs_root,
                tensorboard_root,
                eval_root,
            ):
                folder.mkdir(parents=True, exist_ok=True)

            current_nominal = 0
            model = None

            # Resume from the latest contiguous compatible checkpoint.
            for cp in nominal_checkpoints:
                cp_dir = checkpoints_root / f"step_{cp:06d}"
                model_file = cp_dir / "maskable_ppo.zip"
                metadata_file = cp_dir / "metadata.json"
                if model_file.exists() and metadata_file.exists():
                    meta = load_model_metadata(model_file)
                    validate_model_metadata(
                        meta,
                        generation=args.generation,
                        expected_representation_version=args.ubr,
                    )
                    if meta.seed != int(seed) or meta.total_timesteps != cp:
                        raise ValueError(
                            f"Checkpoint metadata mismatch at {cp_dir}."
                        )
                    if meta.policy_architecture != architecture:
                        raise ValueError(
                            f"Checkpoint architecture mismatch at {cp_dir}: "
                            f"{meta.policy_architecture} != {architecture}."
                        )
                    current_nominal = cp
                else:
                    break

            if current_nominal:
                cp_dir = checkpoints_root / f"step_{current_nominal:06d}"
                model = MaskablePPO.load(
                    str(cp_dir / "maskable_ppo.zip"),
                    device=args.device,
                )
                print(
                    f"[RESUME] {architecture} seed={seed}: "
                    f"checkpoint {current_nominal}."
                )

            for cp in nominal_checkpoints:
                cp_dir = checkpoints_root / f"step_{cp:06d}"

                if cp > current_nominal:
                    previous_nominal = cp - args.checkpoint_interval
                    opponent_key = opponent_at(stages, previous_nominal)
                    monitor_prefix = (
                        logs_root
                        / f"segment_{previous_nominal:06d}_{cp:06d}_{opponent_key}.csv"
                    )
                    env = _make_env(
                        battle_format=battle_format,
                        opponent_key=opponent_key,
                        representation_version=args.ubr,
                        architecture=architecture,
                        reward=reward,
                        log_level=args.log_level,
                        monitor_prefix=monitor_prefix,
                    )

                    if model is None:
                        model = _new_model(
                            env=env,
                            architecture=architecture,
                            base_dim=schema.observation_dim,
                            seed=int(seed),
                            args=args,
                            algo=algo,
                            tensorboard_dir=tensorboard_root,
                        )
                        print(
                            f"[ARCH] {architecture}: family={spec.family}, "
                            f"input={spec.model_input_dim(schema.observation_dim)}, "
                            f"params={_parameter_count(model):,}"
                        )
                    else:
                        model.set_env(env)

                    print(
                        f"[TRAIN] {architecture} seed={seed}: nominal "
                        f"{previous_nominal}->{cp} vs {opponent_key}"
                    )
                    try:
                        model.learn(
                            total_timesteps=int(args.checkpoint_interval),
                            reset_num_timesteps=False,
                            progress_bar=False,
                            tb_log_name=architecture,
                        )
                        _save_checkpoint(
                            model=model,
                            checkpoint_dir=cp_dir,
                            generation=args.generation,
                            representation_version=args.ubr,
                            architecture=architecture,
                            nominal_checkpoint=cp,
                            seed=int(seed),
                            reward=reward,
                            stages=stages,
                            args=args,
                        )
                    finally:
                        env.close()

                    segment_rows = _read_monitor(monitor_prefix)
                    stats = _training_stats(segment_rows)
                    all_training_rows.append({
                        "architecture": architecture,
                        "seed": int(seed),
                        "checkpoint": cp,
                        "training_opponent": opponent_key,
                        "requested_segment_timesteps": int(args.checkpoint_interval),
                        "actual_model_num_timesteps": int(model.num_timesteps),
                        "episodes_segment": stats["episodes"],
                        "mean_reward_segment": stats["mean_reward"],
                        "last_window_reward_segment": stats["last_window_reward"],
                        "mean_episode_length_segment": stats["mean_episode_length"],
                        "source": "trained_v0.5.5",
                        "monitor_file": str(
                            _monitor_path(monitor_prefix) or monitor_prefix
                        ),
                    })
                    current_nominal = cp
                else:
                    previous_nominal = cp - args.checkpoint_interval
                    opponent_key = opponent_at(stages, previous_nominal)
                    monitor_prefix = (
                        logs_root
                        / f"segment_{previous_nominal:06d}_{cp:06d}_{opponent_key}.csv"
                    )
                    segment_rows = _read_monitor(monitor_prefix)
                    stats = _training_stats(segment_rows)
                    architecture_meta_path = cp_dir / "architecture_metadata.json"
                    actual = ""
                    if architecture_meta_path.exists():
                        actual = json.loads(
                            architecture_meta_path.read_text(encoding="utf-8")
                        ).get("actual_model_num_timesteps", "")
                    all_training_rows.append({
                        "architecture": architecture,
                        "seed": int(seed),
                        "checkpoint": cp,
                        "training_opponent": opponent_key,
                        "requested_segment_timesteps": int(args.checkpoint_interval),
                        "actual_model_num_timesteps": actual,
                        "episodes_segment": stats["episodes"],
                        "mean_reward_segment": stats["mean_reward"],
                        "last_window_reward_segment": stats["last_window_reward"],
                        "mean_episode_length_segment": stats["mean_episode_length"],
                        "source": "trained_v0.5.5",
                        "monitor_file": str(
                            _monitor_path(monitor_prefix) or monitor_prefix
                        ),
                    })

                if not (cp_dir / "maskable_ppo.zip").exists():
                    raise RuntimeError(
                        f"Missing checkpoint model: {cp_dir}"
                    )

                kind = "final" if cp == args.total_timesteps else "diagnostic"
                repeats = (
                    args.final_repeats
                    if kind == "final"
                    else args.checkpoint_repeats
                )
                for opponent in args.eval_opponents:
                    all_eval_rows.append(
                        _evaluate_checkpoint(
                            model_dir=cp_dir,
                            eval_root=eval_root,
                            architecture=architecture,
                            seed=int(seed),
                            checkpoint=cp,
                            opponent=opponent,
                            kind=kind,
                            repeats=repeats,
                            args=args,
                        )
                    )

                _write_aggregates(
                    study_root,
                    all_eval_rows,
                    all_training_rows,
                )

    _write_aggregates(study_root, all_eval_rows, all_training_rows)

    print("\n=== PokéBench30 v0.5.5 policy architecture study complete ===")
    print(f"Study: {study_root}")
    print(f"Representation: {schema.version} ({schema.observation_dim} UBR features)")
    print(f"Final summary: {study_root / 'aggregate' / 'final_summary.csv'}")
    print(
        "Architecture comparison: "
        f"{study_root / 'aggregate' / 'architecture_comparison.csv'}"
    )
    return study_root


def build_parser() -> argparse.ArgumentParser:
    config = load_config(DEFAULT_CONFIG)
    block = config["policy_study"]
    algo = config["algorithm"]

    parser = argparse.ArgumentParser(
        description=(
            "PokéBench30 v0.5.5 policy architecture study: fixed UBR 1.2, "
            "reward and Random->TypeAware curriculum; vary only policy architecture."
        )
    )
    parser.add_argument("--generation", type=int, choices=[1, 4, 9], default=9)
    parser.add_argument(
        "--ubr",
        choices=["ubr-1.1", "ubr-1.2"],
        default=block.get("representation", "ubr-1.2"),
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=sorted(DEFAULT_POLICY_ARCHITECTURES),
        default=block.get("architectures", DEFAULT_POLICY_ARCHITECTURES),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=block.get("seeds", [42, 123, 456, 789, 2026]),
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=int(block.get("total_timesteps", 300000)),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=int(block.get("checkpoint_interval", 50000)),
    )
    parser.add_argument(
        "--checkpoint-repeats",
        type=int,
        default=int(block.get("checkpoint_repeats_per_leg", 1)),
    )
    parser.add_argument(
        "--final-repeats",
        type=int,
        default=int(block.get("final_repeats_per_leg", 10)),
    )
    parser.add_argument(
        "--eval-opponents",
        nargs="+",
        default=block.get(
            "evaluation_opponents",
            ["random", "max_damage", "type_aware"],
        ),
    )
    parser.add_argument(
        "--baseline-curriculum-dir",
        default=None,
        help=(
            "Optional completed v0.5.4 curriculum study. Its "
            "random_to_type_aware regime is imported as mlp_baseline."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(algo.get("learning_rate", 3e-4)),
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=int(algo.get("n_steps", 2048)),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(algo.get("batch_size", 128)),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pool-root", default=str(DEFAULT_POOL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--study-dir",
        default=None,
        help="Resume an existing policy-study directory.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "One-seed 6144-step end-to-end test of all selected architectures. "
            "Smoke results are not scientifically interpretable."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_architectures(args.architectures)

    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")
    if args.checkpoint_repeats <= 0 or args.final_repeats <= 0:
        raise ValueError("Evaluation repeats must be > 0.")

    run_study(args)


if __name__ == "__main__":
    main()
