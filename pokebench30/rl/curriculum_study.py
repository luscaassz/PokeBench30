from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import yaml

from pokebench30 import __version__
from pokebench30.representation.schema import build_feature_schema
from pokebench30.rl.model_io import ModelMetadata, load_model_metadata, validate_model_metadata
from pokebench30.rl.reward import RewardConfig
from pokebench30.teams.fixed_pool import DEFAULT_POOL_ROOT, load_pool_manifest, verify_pool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "rl_baseline.yaml"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "curriculum_results"


@dataclass(frozen=True)
class Stage:
    opponent: str
    timesteps: int


@dataclass(frozen=True)
class EvalRow:
    regime: str
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
    model_dir: str
    evaluation_dir: str


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_regimes(config: dict[str, Any]) -> dict[str, list[Stage]]:
    raw = config["curriculum_study"]["regimes"]
    regimes: dict[str, list[Stage]] = {}
    for name, stages in raw.items():
        regimes[name] = [
            Stage(opponent=str(stage["opponent"]), timesteps=int(stage["timesteps"]))
            for stage in stages
        ]
    return regimes


def validate_regimes(
    regimes: dict[str, list[Stage]],
    *,
    total_timesteps: int,
    checkpoint_interval: int,
) -> None:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be > 0")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be > 0")
    if total_timesteps % checkpoint_interval != 0:
        raise ValueError(
            "For v0.5.4, total_timesteps must be divisible by checkpoint_interval "
            "so all regimes are evaluated on identical nominal checkpoints."
        )

    known = {"random", "max_damage", "type_aware", "switch_aware"}
    for name, stages in regimes.items():
        if not stages:
            raise ValueError(f"Regime {name!r} has no stages.")
        total = sum(stage.timesteps for stage in stages)
        if total != total_timesteps:
            raise ValueError(
                f"Regime {name!r} totals {total} timesteps, expected {total_timesteps}."
            )
        cumulative = 0
        for stage in stages:
            if stage.opponent not in known:
                raise ValueError(
                    f"Regime {name!r} uses unknown opponent {stage.opponent!r}."
                )
            if stage.timesteps <= 0:
                raise ValueError(f"Regime {name!r} contains a non-positive stage.")
            cumulative += stage.timesteps
            if cumulative != total_timesteps and cumulative % checkpoint_interval != 0:
                raise ValueError(
                    f"Regime {name!r} changes opponent at {cumulative}, which is not "
                    f"aligned to checkpoint_interval={checkpoint_interval}."
                )


def opponent_at(regime: list[Stage], nominal_step: int) -> str:
    """Opponent used for the interval beginning at nominal_step."""
    if nominal_step < 0:
        raise ValueError("nominal_step must be >= 0")
    cumulative = 0
    for stage in regime:
        end = cumulative + stage.timesteps
        if nominal_step < end:
            return stage.opponent
        cumulative = end
    raise ValueError(
        f"nominal_step={nominal_step} is outside a regime totaling {cumulative}."
    )


def checkpoints(total_timesteps: int, interval: int) -> list[int]:
    return list(range(interval, total_timesteps + 1, interval))


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


def _make_env(
    *,
    battle_format: str,
    opponent_key: str,
    representation_version: str,
    reward: RewardConfig,
    log_level: int,
    monitor_prefix: Path,
):
    try:
        from poke_env.environment import SingleAgentWrapper
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "RL dependencies are not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    from pokebench30.rl.env import UniversalRLSinglesEnv
    from pokebench30.rl.mask_adapter import SB3MaskAdapter
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
    return SB3MaskAdapter(monitored)


def _new_model(
    *,
    env,
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

    policy_kwargs = {"net_arch": list(algo.get("policy_net_arch", [256, 128]))}
    return MaskablePPO(
        algo.get("policy", "MlpPolicy"),
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


def _checkpoint_metadata(
    *,
    generation: int,
    representation_version: str,
    regime_name: str,
    nominal_checkpoint: int,
    actual_num_timesteps: int,
    seed: int,
    reward: RewardConfig,
    stages: list[Stage],
    args: argparse.Namespace,
) -> tuple[ModelMetadata, dict[str, Any]]:
    schema = build_feature_schema(representation_version)
    battle_format = f"gen{generation}randombattle"

    metadata = ModelMetadata(
        project_version=__version__,
        algorithm="MaskablePPO",
        training_generation=generation,
        training_battle_format=battle_format,
        training_opponent=f"curriculum:{regime_name}",
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
            "PokéBench30 v0.5.4 curriculum-training checkpoint.",
            f"Regime: {regime_name}.",
            "Training uses random-battle team generation for diversity.",
            "The reward and UBR are held fixed across curriculum regimes.",
            "Invalid actions are excluded by MaskablePPO action masking.",
            "Nominal checkpoints are labels; PPO collects full n_steps rollouts and may overshoot them.",
        ],
    )
    curriculum = {
        "project": "PokeBench30",
        "version": __version__,
        "experiment_type": "curriculum_training_study",
        "regime": regime_name,
        "stages": [asdict(stage) for stage in stages],
        "nominal_checkpoint_timesteps": int(nominal_checkpoint),
        "actual_model_num_timesteps": int(actual_num_timesteps),
        "seed": int(seed),
        "representation_version": representation_version,
        "learning_rate": float(args.learning_rate),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
    }
    return metadata, curriculum


def _save_checkpoint(
    *,
    model,
    checkpoint_dir: Path,
    generation: int,
    representation_version: str,
    regime_name: str,
    nominal_checkpoint: int,
    seed: int,
    reward: RewardConfig,
    stages: list[Stage],
    args: argparse.Namespace,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_dir / "maskable_ppo"))
    metadata, curriculum = _checkpoint_metadata(
        generation=generation,
        representation_version=representation_version,
        regime_name=regime_name,
        nominal_checkpoint=nominal_checkpoint,
        actual_num_timesteps=int(model.num_timesteps),
        seed=seed,
        reward=reward,
        stages=stages,
        args=args,
    )
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (checkpoint_dir / "curriculum_metadata.json").write_text(
        json.dumps(curriculum, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_eval(
    *,
    regime: str,
    seed: int,
    checkpoint: int,
    opponent: str,
    kind: str,
    repeats: int,
    model_dir: Path,
    eval_dir: Path,
) -> EvalRow:
    rows = _read_csv(eval_dir / "controlled_summary.csv")
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one controlled_summary row at {eval_dir}, got {len(rows)}."
        )
    row = rows[0]
    return EvalRow(
        regime=regime,
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
        model_dir=str(model_dir),
        evaluation_dir=str(eval_dir),
    )


def _evaluate_checkpoint(
    *,
    model_dir: Path,
    eval_root: Path,
    regime_name: str,
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
        regime=regime_name,
        seed=seed,
        checkpoint=checkpoint,
        opponent=opponent,
        kind=kind,
        repeats=repeats,
        model_dir=model_dir,
        eval_dir=eval_dir,
    )


def _write_aggregates(
    root: Path,
    eval_rows: list[EvalRow],
    training_rows: list[dict[str, Any]],
    direct_regime: str = "direct_type_aware",
) -> None:
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)

    eval_dicts = [asdict(row) for row in eval_rows]
    eval_fields = list(asdict(eval_rows[0]).keys()) if eval_rows else [
        "regime", "seed", "checkpoint", "opponent", "evaluation_kind",
        "repeats_per_leg", "battles", "mirror_cycles", "wins_rl",
        "wins_opponent", "draws", "raw_win_rate_rl", "mirror_score_rl",
        "team_assignment_gap", "model_dir", "evaluation_dir",
    ]
    _write_csv(aggregate / "learning_curve.csv", eval_dicts, eval_fields)

    training_fields = [
        "regime", "seed", "checkpoint", "training_opponent",
        "requested_segment_timesteps", "actual_model_num_timesteps",
        "episodes_segment", "mean_reward_segment", "last_window_reward_segment",
        "mean_episode_length_segment", "monitor_file",
    ]
    _write_csv(aggregate / "training_progress.csv", training_rows, training_fields)

    grouped: dict[tuple[str, int, str, str], list[EvalRow]] = {}
    for row in eval_rows:
        grouped.setdefault(
            (row.regime, row.checkpoint, row.opponent, row.evaluation_kind), []
        ).append(row)

    agg_rows: list[dict[str, Any]] = []
    for (regime, checkpoint, opponent, kind), rows in sorted(grouped.items()):
        scores = [row.mirror_score_rl for row in rows]
        agg_rows.append({
            "regime": regime,
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
            "regime", "checkpoint", "opponent", "evaluation_kind", "n_seeds",
            "mean_mirror_score", "sample_sd_mirror_score", "min_mirror_score",
            "max_mirror_score", "total_wins_rl", "total_wins_opponent",
            "total_draws", "total_battles",
        ],
    )

    final_rows = [row for row in eval_rows if row.evaluation_kind == "final"]
    final_grouped: dict[tuple[str, str], list[EvalRow]] = {}
    for row in final_rows:
        final_grouped.setdefault((row.regime, row.opponent), []).append(row)

    final_summary: list[dict[str, Any]] = []
    for (regime, opponent), rows in sorted(final_grouped.items()):
        scores = [row.mirror_score_rl for row in rows]
        final_summary.append({
            "regime": regime,
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
            "regime", "opponent", "n_seeds", "mean_mirror_score",
            "sample_sd_mirror_score", "min_mirror_score", "max_mirror_score",
            "total_wins_rl", "total_wins_opponent", "total_draws", "total_battles",
        ],
    )

    by_key = {(row.regime, row.seed, row.opponent): row for row in final_rows}
    regimes = sorted({row.regime for row in final_rows})
    seeds = sorted({row.seed for row in final_rows})
    opponents = sorted({row.opponent for row in final_rows})
    pair_rows: list[dict[str, Any]] = []
    for regime in regimes:
        if regime == direct_regime:
            continue
        for opponent in opponents:
            for seed in seeds:
                baseline = by_key.get((direct_regime, seed, opponent))
                candidate = by_key.get((regime, seed, opponent))
                if baseline is None or candidate is None:
                    continue
                pair_rows.append({
                    "regime": regime,
                    "seed": seed,
                    "opponent": opponent,
                    "direct_score": baseline.mirror_score_rl,
                    "regime_score": candidate.mirror_score_rl,
                    "difference_vs_direct": candidate.mirror_score_rl - baseline.mirror_score_rl,
                })
    _write_csv(
        aggregate / "paired_regime_differences.csv",
        pair_rows,
        [
            "regime", "seed", "opponent", "direct_score", "regime_score",
            "difference_vs_direct",
        ],
    )

    comparison_rows: list[dict[str, Any]] = []
    for regime in regimes:
        if regime == direct_regime:
            continue
        for opponent in opponents:
            rows = [
                row for row in pair_rows
                if row["regime"] == regime and row["opponent"] == opponent
            ]
            if not rows:
                continue
            diffs = [float(row["difference_vs_direct"]) for row in rows]
            comparison_rows.append({
                "regime": regime,
                "opponent": opponent,
                "paired_seeds": len(rows),
                "mean_direct": statistics.fmean(float(row["direct_score"]) for row in rows),
                "mean_regime": statistics.fmean(float(row["regime_score"]) for row in rows),
                "mean_difference_vs_direct": statistics.fmean(diffs),
                "sample_sd_difference": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
            })
    _write_csv(
        aggregate / "curriculum_comparison.csv",
        comparison_rows,
        [
            "regime", "opponent", "paired_seeds", "mean_direct", "mean_regime",
            "mean_difference_vs_direct", "sample_sd_difference",
        ],
    )

    figures = aggregate / "figures"
    figures.mkdir(exist_ok=True)
    for opponent in sorted({row.opponent for row in eval_rows}):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for regime in sorted({row.regime for row in eval_rows}):
            points = [
                row for row in agg_rows
                if row["regime"] == regime and row["opponent"] == opponent
            ]
            points.sort(key=lambda x: int(x["checkpoint"]))
            if not points:
                continue
            ax.errorbar(
                [int(row["checkpoint"]) for row in points],
                [float(row["mean_mirror_score"]) for row in points],
                yerr=[float(row["sample_sd_mirror_score"]) for row in points],
                marker="o",
                capsize=3,
                label=regime,
            )
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Nominal training timesteps")
        ax.set_ylabel("Mean mirrored score (± seed SD)")
        ax.set_title(f"Curriculum learning curves vs {opponent}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"learning_curve_vs_{opponent}.png", dpi=160)
        plt.close(fig)


def _study_manifest(args: argparse.Namespace, regimes: dict[str, list[Stage]], pool: dict[str, Any]) -> dict[str, Any]:
    return {
        "project": "PokeBench30",
        "version": __version__,
        "experiment_type": "curriculum_training_study",
        "generation": int(args.generation),
        "representation": args.ubr,
        "reward_profile": "v0.5_dense_reward_unchanged",
        "seeds": [int(x) for x in args.seeds],
        "total_timesteps": int(args.total_timesteps),
        "checkpoint_interval": int(args.checkpoint_interval),
        "checkpoint_repeats_per_leg": int(args.checkpoint_repeats),
        "final_repeats_per_leg": int(args.final_repeats),
        "evaluation_opponents": list(args.eval_opponents),
        "regimes": {
            name: [asdict(stage) for stage in stages]
            for name, stages in regimes.items()
        },
        "ppo": {
            "learning_rate": float(args.learning_rate),
            "learning_rate_schedule": "constant",
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
        },
        "fixed_team_pool": {
            "config_sha256": pool.get("config_sha256"),
            "showdown_commit": pool.get("showdown_commit"),
            "showdown_dirty": pool.get("showdown_dirty"),
        },
        "design_notes": [
            "v0.5.4 isolates training-opponent curriculum: UBR, reward, PPO architecture and hyperparameters remain fixed.",
            "All regimes receive the same nominal total training budget.",
            "Intermediate checkpoint evaluations are diagnostic and intentionally small.",
            "Final evaluations use the full fixed/mirrored protocol.",
            "PPO collects full rollouts; actual model.num_timesteps may exceed nominal checkpoints.",
            "Pokemon Showdown battle RNG is not common-random-number paired across regimes.",
        ],
    }


def _same_manifest(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = [
        "generation", "representation", "seeds", "total_timesteps",
        "checkpoint_interval", "checkpoint_repeats_per_leg",
        "final_repeats_per_leg", "evaluation_opponents", "regimes", "ppo",
        "fixed_team_pool",
    ]
    return all(a.get(key) == b.get(key) for key in keys)


def run_study(args: argparse.Namespace) -> Path:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "RL dependencies are not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    config = load_config(Path(args.config))
    algo = config["algorithm"]
    reward = RewardConfig.from_mapping(config.get("reward"))
    all_regimes = parse_regimes(config)

    if args.smoke:
        # Small end-to-end test of all four regime transitions. It is not
        # intended for scientific interpretation.
        args.seeds = [42]
        args.total_timesteps = 6144
        args.checkpoint_interval = 2048
        args.checkpoint_repeats = 1
        args.final_repeats = 1
        args.n_steps = 256
        args.batch_size = min(int(args.batch_size), 64)
        all_regimes = {
            "direct_type_aware": [Stage("type_aware", 6144)],
            "random_to_type_aware": [
                Stage("random", 2048),
                Stage("type_aware", 4096),
            ],
            "progressive": [
                Stage("random", 2048),
                Stage("max_damage", 2048),
                Stage("type_aware", 2048),
            ],
            "random_only": [Stage("random", 6144)],
        }
        print("[SMOKE] Using 1 seed, 6,144 total timesteps, 2,048-step checkpoints.")
    selected_names = list(args.regimes) if args.regimes else list(all_regimes)
    missing = [name for name in selected_names if name not in all_regimes]
    if missing:
        raise ValueError(f"Unknown curriculum regime(s): {missing}")
    regimes = {name: all_regimes[name] for name in selected_names}

    validate_regimes(
        regimes,
        total_timesteps=args.total_timesteps,
        checkpoint_interval=args.checkpoint_interval,
    )

    pool_root = Path(args.pool_root).resolve()
    problems = verify_pool(pool_root)
    if problems:
        raise RuntimeError(
            "Fixed-team pool failed integrity check:\n- " + "\n- ".join(problems)
        )
    pool_manifest = load_pool_manifest(pool_root)

    study_root = (
        Path(args.study_dir).resolve()
        if args.study_dir
        else Path(args.output_root).resolve()
        / f"curriculum_gen{args.generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    study_root.mkdir(parents=True, exist_ok=True)
    manifest = _study_manifest(args, regimes, pool_manifest)
    manifest_path = study_root / "study_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _same_manifest(previous, manifest):
            raise ValueError(
                "The selected --study-dir contains a different curriculum-study configuration."
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    schema = build_feature_schema(args.ubr)
    battle_format = f"gen{args.generation}randombattle"
    nominal_checkpoints = checkpoints(args.total_timesteps, args.checkpoint_interval)

    all_eval_rows: list[EvalRow] = []
    all_training_rows: list[dict[str, Any]] = []
    total_tracks = len(regimes) * len(args.seeds)
    track_index = 0

    for regime_name, stages in regimes.items():
        for seed in args.seeds:
            track_index += 1
            run_root = study_root / "runs" / regime_name / f"seed_{int(seed)}"
            checkpoints_root = run_root / "checkpoints"
            eval_root = run_root / "evaluations"
            logs_root = run_root / "training_logs"
            tensorboard_root = run_root / "tensorboard"
            for path in (checkpoints_root, eval_root, logs_root, tensorboard_root):
                path.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 82)
            print(
                f"TRACK {track_index}/{total_tracks}: {regime_name} | seed={seed} | "
                f"{args.ubr} | {args.total_timesteps} nominal timesteps"
            )
            print("=" * 82)

            # Locate highest contiguous model checkpoint for resume.
            current_nominal = 0
            model = None
            for cp in nominal_checkpoints:
                cp_dir = checkpoints_root / f"step_{cp:06d}"
                if (cp_dir / "maskable_ppo.zip").exists() and (cp_dir / "metadata.json").exists():
                    meta = load_model_metadata(cp_dir / "maskable_ppo.zip")
                    validate_model_metadata(
                        meta,
                        generation=args.generation,
                        expected_representation_version=args.ubr,
                    )
                    if meta.seed != int(seed) or meta.total_timesteps != cp:
                        raise ValueError(f"Checkpoint metadata mismatch at {cp_dir}")
                    current_nominal = cp
                else:
                    break

            if current_nominal:
                cp_dir = checkpoints_root / f"step_{current_nominal:06d}"
                model = MaskablePPO.load(str(cp_dir / "maskable_ppo.zip"), device=args.device)
                print(f"[RESUME] Loaded checkpoint {current_nominal}: {cp_dir}")

            # Ensure evaluations for existing checkpoints are present, then train onward.
            for cp in nominal_checkpoints:
                cp_dir = checkpoints_root / f"step_{cp:06d}"

                if cp > current_nominal:
                    previous_nominal = cp - args.checkpoint_interval
                    opponent_key = opponent_at(stages, previous_nominal)
                    monitor_prefix = logs_root / f"segment_{previous_nominal:06d}_{cp:06d}_{opponent_key}.csv"
                    env = _make_env(
                        battle_format=battle_format,
                        opponent_key=opponent_key,
                        representation_version=args.ubr,
                        reward=reward,
                        log_level=args.log_level,
                        monitor_prefix=monitor_prefix,
                    )

                    if model is None:
                        model = _new_model(
                            env=env,
                            seed=int(seed),
                            args=args,
                            algo=algo,
                            tensorboard_dir=tensorboard_root,
                        )
                    else:
                        model.set_env(env)

                    print(
                        f"[TRAIN] {regime_name} seed={seed}: nominal "
                        f"{previous_nominal}->{cp} vs {opponent_key}"
                    )
                    try:
                        model.learn(
                            total_timesteps=int(args.checkpoint_interval),
                            reset_num_timesteps=False,
                            progress_bar=False,
                            tb_log_name=regime_name,
                        )
                        _save_checkpoint(
                            model=model,
                            checkpoint_dir=cp_dir,
                            generation=args.generation,
                            representation_version=args.ubr,
                            regime_name=regime_name,
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
                        "regime": regime_name,
                        "seed": int(seed),
                        "checkpoint": cp,
                        "training_opponent": opponent_key,
                        "requested_segment_timesteps": int(args.checkpoint_interval),
                        "actual_model_num_timesteps": int(model.num_timesteps),
                        "episodes_segment": stats["episodes"],
                        "mean_reward_segment": stats["mean_reward"],
                        "last_window_reward_segment": stats["last_window_reward"],
                        "mean_episode_length_segment": stats["mean_episode_length"],
                        "monitor_file": str(_monitor_path(monitor_prefix) or monitor_prefix),
                    })
                    current_nominal = cp
                else:
                    # Reconstruct training row on resume if monitor exists.
                    previous_nominal = cp - args.checkpoint_interval
                    opponent_key = opponent_at(stages, previous_nominal)
                    monitor_prefix = logs_root / f"segment_{previous_nominal:06d}_{cp:06d}_{opponent_key}.csv"
                    segment_rows = _read_monitor(monitor_prefix)
                    stats = _training_stats(segment_rows)
                    curriculum_meta_path = cp_dir / "curriculum_metadata.json"
                    actual = ""
                    if curriculum_meta_path.exists():
                        actual = json.loads(curriculum_meta_path.read_text(encoding="utf-8")).get(
                            "actual_model_num_timesteps", ""
                        )
                    all_training_rows.append({
                        "regime": regime_name,
                        "seed": int(seed),
                        "checkpoint": cp,
                        "training_opponent": opponent_key,
                        "requested_segment_timesteps": int(args.checkpoint_interval),
                        "actual_model_num_timesteps": actual,
                        "episodes_segment": stats["episodes"],
                        "mean_reward_segment": stats["mean_reward"],
                        "last_window_reward_segment": stats["last_window_reward"],
                        "mean_episode_length_segment": stats["mean_episode_length"],
                        "monitor_file": str(_monitor_path(monitor_prefix) or monitor_prefix),
                    })

                if not (cp_dir / "maskable_ppo.zip").exists():
                    raise RuntimeError(f"Missing checkpoint model: {cp_dir}")

                # Intermediate checkpoints use a small diagnostic sample; the final
                # checkpoint uses the full publication-style evaluation sample.
                kind = "final" if cp == args.total_timesteps else "diagnostic"
                repeats = args.final_repeats if kind == "final" else args.checkpoint_repeats
                for opponent in args.eval_opponents:
                    all_eval_rows.append(
                        _evaluate_checkpoint(
                            model_dir=cp_dir,
                            eval_root=eval_root,
                            regime_name=regime_name,
                            seed=int(seed),
                            checkpoint=cp,
                            opponent=opponent,
                            kind=kind,
                            repeats=repeats,
                            args=args,
                        )
                    )

                _write_aggregates(study_root, all_eval_rows, all_training_rows)

    _write_aggregates(study_root, all_eval_rows, all_training_rows)
    print("\n=== PokéBench30 v0.5.4 curriculum study complete ===")
    print(f"Study: {study_root}")
    print(f"Representation: {schema.version} ({schema.observation_dim} features)")
    print(f"Final summary: {study_root / 'aggregate' / 'final_summary.csv'}")
    print(f"Curriculum comparison: {study_root / 'aggregate' / 'curriculum_comparison.csv'}")
    return study_root


def build_parser() -> argparse.ArgumentParser:
    config = load_config(DEFAULT_CONFIG)
    block = config["curriculum_study"]
    algo = config["algorithm"]

    parser = argparse.ArgumentParser(
        description=(
            "PokéBench30 v0.5.4 curriculum study: fixed UBR/reward/PPO, "
            "vary only the sequence of training opponents."
        )
    )
    parser.add_argument("--generation", type=int, choices=[1, 4, 9], default=9)
    parser.add_argument("--ubr", choices=["ubr-1.1", "ubr-1.2"], default=block.get("representation", "ubr-1.2"))
    parser.add_argument("--seeds", nargs="+", type=int, default=block.get("seeds", [42, 123, 456, 789, 2026]))
    parser.add_argument("--regimes", nargs="+", default=None, help="Subset of configured regime names. Default: all.")
    parser.add_argument("--total-timesteps", type=int, default=int(block.get("total_timesteps", 300000)))
    parser.add_argument("--checkpoint-interval", type=int, default=int(block.get("checkpoint_interval", 50000)))
    parser.add_argument("--checkpoint-repeats", type=int, default=int(block.get("checkpoint_repeats_per_leg", 1)))
    parser.add_argument("--final-repeats", type=int, default=int(block.get("final_repeats_per_leg", 10)))
    parser.add_argument("--eval-opponents", nargs="+", default=block.get("evaluation_opponents", ["random", "max_damage", "type_aware"]))
    parser.add_argument("--learning-rate", type=float, default=float(algo.get("learning_rate", 3e-4)))
    parser.add_argument("--n-steps", type=int, default=int(algo.get("n_steps", 2048)))
    parser.add_argument("--batch-size", type=int, default=int(algo.get("batch_size", 128)))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pool-root", default=str(DEFAULT_POOL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--study-dir", default=None, help="Resume an existing study directory.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a small one-seed end-to-end test of all curriculum regimes. "
            "Smoke results are not scientifically interpretable."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")
    if args.checkpoint_repeats <= 0 or args.final_repeats <= 0:
        raise ValueError("Evaluation repeats must be > 0.")
    if args.n_steps <= 0 or args.batch_size <= 0:
        raise ValueError("n_steps and batch_size must be > 0.")
    run_study(args)


if __name__ == "__main__":
    main()
