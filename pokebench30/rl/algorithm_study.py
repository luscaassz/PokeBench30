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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pokebench30 import __version__
from pokebench30.representation.schema import build_feature_schema
from pokebench30.rl.masked_dqn import DQNConfig, MaskedDQNTrainer
from pokebench30.rl.model_io import ModelMetadata, load_model_metadata, validate_model_metadata
from pokebench30.rl.reward import RewardConfig
from pokebench30.teams.fixed_pool import DEFAULT_POOL_ROOT, load_pool_manifest, verify_pool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "rl_baseline.yaml"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "algorithm_results"

ALGORITHM_KEYS = ("maskable_ppo", "masked_dqn", "masked_double_dqn")
ALGORITHM_METADATA_NAMES = {
    "maskable_ppo": "MaskablePPO",
    "masked_dqn": "MaskedDQN",
    "masked_double_dqn": "MaskedDoubleDQN",
}


@dataclass(frozen=True)
class Stage:
    opponent: str
    timesteps: int


@dataclass(frozen=True)
class EvalRow:
    algorithm: str
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


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False))


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


def _remove_partial_monitor(prefix: Path) -> None:
    candidates = {
        prefix,
        Path(str(prefix) + ".monitor.csv"),
        prefix.with_suffix(prefix.suffix + ".monitor.csv") if prefix.suffix else prefix,
    }
    for candidate in candidates:
        if candidate.exists():
            candidate.unlink()


def parse_curriculum(config: dict[str, Any]) -> list[Stage]:
    return [
        Stage(opponent=str(item["opponent"]), timesteps=int(item["timesteps"]))
        for item in config["algorithm_study"]["curriculum"]
    ]


def validate_curriculum(
    stages: list[Stage],
    *,
    total_timesteps: int,
    checkpoint_interval: int,
) -> None:
    if not stages:
        raise ValueError("Algorithm-study curriculum cannot be empty.")
    if sum(stage.timesteps for stage in stages) != int(total_timesteps):
        raise ValueError("Curriculum total does not match total_timesteps.")
    cumulative = 0
    for stage in stages:
        if stage.timesteps <= 0:
            raise ValueError("Curriculum stage timesteps must be > 0.")
        cumulative += stage.timesteps
        if (
            cumulative != int(total_timesteps)
            and cumulative % int(checkpoint_interval) != 0
        ):
            raise ValueError(
                f"Curriculum transition {cumulative} is not checkpoint-aligned."
            )


def opponent_at(stages: list[Stage], nominal_step: int) -> str:
    cumulative = 0
    for stage in stages:
        end = cumulative + stage.timesteps
        if nominal_step < end:
            return stage.opponent
        cumulative = end
    raise ValueError(f"nominal_step={nominal_step} is outside the curriculum.")


def nominal_checkpoints(total_timesteps: int, interval: int) -> list[int]:
    if total_timesteps % interval != 0:
        raise ValueError("total_timesteps must be divisible by checkpoint_interval.")
    return list(range(interval, total_timesteps + 1, interval))


def effective_segment_steps(checkpoint_interval: int, ppo_n_steps: int) -> int:
    """Match the actual environment-interaction count of the PPO baseline."""
    return int(
        math.ceil(int(checkpoint_interval) / int(ppo_n_steps))
        * int(ppo_n_steps)
    )


def make_env(
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
            "RL dependencies are not installed. "
            "Run: python -m pip install -r requirements.txt"
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
    single = SingleAgentWrapper(base_env, opponent)
    monitored = Monitor(single, filename=str(monitor_prefix))
    return SB3MaskAdapter(monitored)


def _metadata(
    *,
    algorithm_key: str,
    generation: int,
    representation_version: str,
    nominal_checkpoint: int,
    actual_environment_steps: int,
    seed: int,
    reward: RewardConfig,
    algorithm_config: dict[str, Any],
) -> ModelMetadata:
    schema = build_feature_schema(representation_version)
    return ModelMetadata(
        project_version=__version__,
        algorithm=ALGORITHM_METADATA_NAMES[algorithm_key],
        training_generation=int(generation),
        training_battle_format=f"gen{generation}randombattle",
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
            "PokéBench30 v0.5.6 RL algorithm study.",
            "UBR 1.2, MLP [256, 128], reward, curriculum, seeds, environment budget and evaluation protocol are fixed.",
            "All algorithms use strict legal-action masking.",
            "DQN targets mask illegal next-state actions before max/argmax.",
            "Double DQN uses the online network for masked next-action selection and the target network for evaluation.",
            f"Nominal checkpoint={nominal_checkpoint}; actual environment steps={actual_environment_steps}.",
        ],
        policy_architecture="mlp_baseline",
        policy_config={
            "family": "mlp",
            "history_len": 1,
            "net_arch": [256, 128],
        },
        model_input_dim=schema.observation_dim,
        algorithm_config=dict(algorithm_config),
    )


def _save_metadata(directory: Path, metadata: ModelMetadata) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_eval(
    *,
    algorithm: str,
    seed: int,
    checkpoint: int,
    opponent: str,
    kind: str,
    repeats: int,
    model_dir: Path,
    eval_dir: Path,
    source: str,
) -> EvalRow:
    rows = _read_csv(eval_dir / "controlled_summary.csv")
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one controlled_summary row in {eval_dir}; got {len(rows)}."
        )
    row = rows[0]
    return EvalRow(
        algorithm=algorithm,
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


def evaluate_checkpoint(
    *,
    algorithm: str,
    seed: int,
    checkpoint: int,
    opponent: str,
    kind: str,
    repeats: int,
    model_dir: Path,
    eval_root: Path,
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
            "--model", str(model_dir),
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
        algorithm=algorithm,
        seed=seed,
        checkpoint=checkpoint,
        opponent=opponent,
        kind=kind,
        repeats=repeats,
        model_dir=model_dir,
        eval_dir=eval_dir,
        source="trained_v0.5.6",
    )


def _validate_baseline_source(
    source_root: Path,
    *,
    args: argparse.Namespace,
    stages: list[Stage],
    pool_manifest: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    manifest_path = source_root / "study_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Baseline study manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    experiment_type = str(manifest.get("experiment_type", ""))
    if experiment_type not in {
        "policy_architecture_study",
        "curriculum_training_study",
    }:
        raise ValueError(
            "Baseline source must be a completed v0.5.5 policy study or "
            "v0.5.4 curriculum study."
        )

    errors: list[str] = []

    def expect(label: str, observed: Any, expected: Any) -> None:
        if _canonical(observed) != _canonical(expected):
            errors.append(
                f"{label}: observed={observed!r}, expected={expected!r}"
            )

    expect("generation", int(manifest.get("generation", -1)), int(args.generation))
    expect("representation", manifest.get("representation"), args.ubr)
    expect("seeds", manifest.get("seeds"), [int(v) for v in args.seeds])
    expect("total_timesteps", int(manifest.get("total_timesteps", -1)), int(args.total_timesteps))
    expect("checkpoint_interval", int(manifest.get("checkpoint_interval", -1)), int(args.checkpoint_interval))
    expect("checkpoint_repeats_per_leg", int(manifest.get("checkpoint_repeats_per_leg", -1)), int(args.checkpoint_repeats))
    expect("final_repeats_per_leg", int(manifest.get("final_repeats_per_leg", -1)), int(args.final_repeats))
    expect("evaluation_opponents", manifest.get("evaluation_opponents"), list(args.eval_opponents))

    expected_curriculum = [asdict(stage) for stage in stages]
    if experiment_type == "policy_architecture_study":
        expect("curriculum", manifest.get("curriculum"), expected_curriculum)
        if "mlp_baseline" not in manifest.get("architectures", []):
            errors.append("v0.5.5 source does not contain mlp_baseline.")
    else:
        observed = (
            manifest.get("regimes", {})
            .get("random_to_type_aware")
        )
        expect("random_to_type_aware curriculum", observed, expected_curriculum)

    ppo = manifest.get("ppo", {})
    if ppo:
        expect("ppo.learning_rate", float(ppo.get("learning_rate", -1.0)), float(args.ppo_learning_rate))
        expect("ppo.n_steps", int(ppo.get("n_steps", -1)), int(args.ppo_n_steps))
        expect("ppo.batch_size", int(ppo.get("batch_size", -1)), int(args.ppo_batch_size))

    pool = manifest.get("fixed_team_pool", {})
    expect("pool.config_sha256", pool.get("config_sha256"), pool_manifest.get("config_sha256"))
    expect("pool.showdown_commit", pool.get("showdown_commit"), pool_manifest.get("showdown_commit"))

    if errors:
        raise ValueError(
            "The supplied PPO baseline source is incompatible:\n- "
            + "\n- ".join(errors)
        )

    return experiment_type, manifest


def import_ppo_baseline(
    source_root: Path,
    *,
    args: argparse.Namespace,
    stages: list[Stage],
    pool_manifest: dict[str, Any],
    snapshot_dir: Path,
) -> tuple[list[EvalRow], list[dict[str, Any]]]:
    source_root = Path(source_root).resolve()
    experiment_type, manifest = _validate_baseline_source(
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
            "Baseline source must contain aggregate/learning_curve.csv and "
            "aggregate/training_progress.csv."
        )

    eval_rows: list[EvalRow] = []
    for row in _read_csv(learning_path):
        if experiment_type == "policy_architecture_study":
            if row["architecture"] != "mlp_baseline":
                continue
        else:
            if row["regime"] != "random_to_type_aware":
                continue

        eval_rows.append(
            EvalRow(
                algorithm="maskable_ppo",
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
                source=f"imported_{experiment_type}",
                model_dir=row["model_dir"],
                evaluation_dir=row["evaluation_dir"],
            )
        )

    training_rows: list[dict[str, Any]] = []
    for row in _read_csv(training_path):
        selector = (
            row.get("architecture") == "mlp_baseline"
            if experiment_type == "policy_architecture_study"
            else row.get("regime") == "random_to_type_aware"
        )
        if not selector:
            continue

        training_rows.append({
            "algorithm": "maskable_ppo",
            "seed": int(row["seed"]),
            "checkpoint": int(row["checkpoint"]),
            "training_opponent": row["training_opponent"],
            "requested_segment_timesteps": int(row["requested_segment_timesteps"]),
            "actual_environment_steps": row["actual_model_num_timesteps"],
            "episodes_segment": row["episodes_segment"],
            "mean_reward_segment": row["mean_reward_segment"],
            "last_window_reward_segment": row["last_window_reward_segment"],
            "mean_episode_length_segment": row["mean_episode_length_segment"],
            "mean_loss_segment": "",
            "epsilon_end": "",
            "gradient_updates": "",
            "replay_size": "",
            "source": f"imported_{experiment_type}",
            "monitor_file": row["monitor_file"],
        })

    expected_eval = (
        len(args.seeds)
        * len(nominal_checkpoints(args.total_timesteps, args.checkpoint_interval))
        * len(args.eval_opponents)
    )
    expected_training = (
        len(args.seeds)
        * len(nominal_checkpoints(args.total_timesteps, args.checkpoint_interval))
    )
    if len(eval_rows) != expected_eval:
        raise RuntimeError(
            f"Expected {expected_eval} PPO evaluation rows, got {len(eval_rows)}."
        )
    if len(training_rows) != expected_training:
        raise RuntimeError(
            f"Expected {expected_training} PPO training rows, got {len(training_rows)}."
        )

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source_root / "study_manifest.json",
        snapshot_dir / "source_study_manifest.json",
    )
    _write_csv(
        snapshot_dir / "learning_curve_maskable_ppo.csv",
        [asdict(row) for row in eval_rows],
        list(asdict(eval_rows[0]).keys()),
    )
    _write_csv(
        snapshot_dir / "training_progress_maskable_ppo.csv",
        training_rows,
        list(training_rows[0].keys()),
    )
    (snapshot_dir / "import_info.json").write_text(
        json.dumps(
            {
                "source_experiment_dir": str(source_root),
                "source_experiment_type": experiment_type,
                "mapped_algorithm": "maskable_ppo",
                "source_manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return eval_rows, training_rows


def _ppo_model(
    *,
    env,
    seed: int,
    args: argparse.Namespace,
    config: dict[str, Any],
    tensorboard_dir: Path,
):
    from sb3_contrib import MaskablePPO

    algo = config["algorithm"]
    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=float(args.ppo_learning_rate),
        n_steps=int(args.ppo_n_steps),
        batch_size=int(args.ppo_batch_size),
        n_epochs=int(algo.get("n_epochs", 10)),
        gamma=float(algo.get("gamma", 0.99)),
        gae_lambda=float(algo.get("gae_lambda", 0.95)),
        clip_range=float(algo.get("clip_range", 0.2)),
        ent_coef=float(algo.get("ent_coef", 0.01)),
        vf_coef=float(algo.get("vf_coef", 0.5)),
        max_grad_norm=float(algo.get("max_grad_norm", 0.5)),
        policy_kwargs={"net_arch": [256, 128]},
        seed=int(seed),
        device=args.device,
        tensorboard_log=str(tensorboard_dir),
        verbose=1,
    )


def _read_monitor_rewards(prefix: Path) -> dict[str, Any]:
    path = _monitor_path(prefix)
    if path is None:
        return {
            "episodes": 0,
            "mean_reward": "",
            "last_window_reward": "",
            "mean_episode_length": "",
        }

    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        if not first.startswith("#"):
            handle.seek(0)
        rows = list(csv.DictReader(handle))

    rewards = [float(row["r"]) for row in rows if row.get("r")]
    lengths = [float(row["l"]) for row in rows if row.get("l")]
    return {
        "episodes": len(rows),
        "mean_reward": statistics.fmean(rewards) if rewards else "",
        "last_window_reward": statistics.fmean(rewards[-20:]) if rewards else "",
        "mean_episode_length": statistics.fmean(lengths) if lengths else "",
    }


def _save_ppo_checkpoint(
    *,
    model,
    directory: Path,
    generation: int,
    representation_version: str,
    checkpoint: int,
    seed: int,
    reward: RewardConfig,
    config: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save(str(directory / "maskable_ppo"))
    algo = config["algorithm"]
    metadata = _metadata(
        algorithm_key="maskable_ppo",
        generation=generation,
        representation_version=representation_version,
        nominal_checkpoint=checkpoint,
        actual_environment_steps=int(model.num_timesteps),
        seed=seed,
        reward=reward,
        algorithm_config={
            "learning_rate": float(algo["learning_rate"]),
            "n_steps": int(algo["n_steps"]),
            "batch_size": int(algo["batch_size"]),
            "n_epochs": int(algo.get("n_epochs", 10)),
            "gamma": float(algo.get("gamma", 0.99)),
            "gae_lambda": float(algo.get("gae_lambda", 0.95)),
            "clip_range": float(algo.get("clip_range", 0.2)),
            "ent_coef": float(algo.get("ent_coef", 0.01)),
            "vf_coef": float(algo.get("vf_coef", 0.5)),
        },
    )
    _save_metadata(directory, metadata)


def _save_dqn_checkpoint(
    *,
    trainer: MaskedDQNTrainer,
    directory: Path,
    algorithm_key: str,
    generation: int,
    representation_version: str,
    checkpoint: int,
    seed: int,
    reward: RewardConfig,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    trainer.save_model(directory / "masked_dqn.pt")
    metadata = _metadata(
        algorithm_key=algorithm_key,
        generation=generation,
        representation_version=representation_version,
        nominal_checkpoint=checkpoint,
        actual_environment_steps=trainer.environment_steps,
        seed=seed,
        reward=reward,
        algorithm_config=trainer.config.to_dict(),
    )
    _save_metadata(directory, metadata)
    (directory / "training_state_summary.json").write_text(
        json.dumps(
            {
                "environment_steps": trainer.environment_steps,
                "gradient_updates": trainer.gradient_updates,
                "epsilon": trainer.epsilon(),
                "replay_size": len(trainer.replay),
                "parameter_count": trainer.parameter_count(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _train_dqn_segment(
    *,
    trainer: MaskedDQNTrainer,
    env,
    steps: int,
) -> dict[str, Any]:
    observation, _ = env.reset()
    action_mask = env.action_masks()

    episode_reward = 0.0
    episode_length = 0
    completed_rewards: list[float] = []
    completed_lengths: list[int] = []
    losses: list[float] = []

    for _ in range(int(steps)):
        action = trainer.select_action(
            observation,
            action_mask,
            deterministic=False,
        )
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        if done:
            next_mask = np.ones(trainer.action_dim, dtype=bool)
        else:
            next_mask = env.action_masks()

        trainer.observe(
            observation=observation,
            action=action,
            reward=float(reward),
            next_observation=next_observation,
            done=done,
            next_action_mask=next_mask,
        )
        losses.extend(trainer.maybe_train())

        episode_reward += float(reward)
        episode_length += 1

        if done:
            completed_rewards.append(episode_reward)
            completed_lengths.append(episode_length)
            observation, _ = env.reset()
            action_mask = env.action_masks()
            episode_reward = 0.0
            episode_length = 0
        else:
            observation = next_observation
            action_mask = next_mask

    return {
        "episodes": len(completed_rewards),
        "mean_reward": (
            statistics.fmean(completed_rewards)
            if completed_rewards else ""
        ),
        "last_window_reward": (
            statistics.fmean(completed_rewards[-20:])
            if completed_rewards else ""
        ),
        "mean_episode_length": (
            statistics.fmean(completed_lengths)
            if completed_lengths else ""
        ),
        "mean_loss": statistics.fmean(losses) if losses else "",
        "epsilon_end": trainer.epsilon(),
        "gradient_updates": trainer.gradient_updates,
        "replay_size": len(trainer.replay),
    }


def _checkpoint_exists(root: Path, algorithm: str, checkpoint: int) -> bool:
    directory = root / f"step_{checkpoint:06d}"
    model_name = (
        "maskable_ppo.zip"
        if algorithm == "maskable_ppo"
        else "masked_dqn.pt"
    )
    return (directory / model_name).exists() and (directory / "metadata.json").exists()


def _latest_contiguous_checkpoint(
    checkpoint_root: Path,
    *,
    algorithm: str,
    checkpoints: list[int],
    generation: int,
    representation_version: str,
    seed: int,
) -> int:
    latest = 0
    for checkpoint in checkpoints:
        directory = checkpoint_root / f"step_{checkpoint:06d}"
        if not _checkpoint_exists(checkpoint_root, algorithm, checkpoint):
            break
        meta = load_model_metadata(directory)
        validate_model_metadata(
            meta,
            generation=generation,
            expected_representation_version=representation_version,
        )
        if int(meta.seed) != int(seed):
            raise ValueError(f"Seed mismatch in checkpoint {directory}.")
        if int(meta.total_timesteps) != int(checkpoint):
            raise ValueError(f"Nominal-step mismatch in checkpoint {directory}.")
        if meta.algorithm != ALGORITHM_METADATA_NAMES[algorithm]:
            raise ValueError(f"Algorithm mismatch in checkpoint {directory}.")
        latest = checkpoint
    return latest


def _write_aggregates(
    study_root: Path,
    eval_rows: list[EvalRow],
    training_rows: list[dict[str, Any]],
) -> None:
    aggregate = study_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)

    eval_fields = list(asdict(eval_rows[0]).keys()) if eval_rows else [
        "algorithm", "seed", "checkpoint", "opponent", "evaluation_kind",
        "repeats_per_leg", "battles", "mirror_cycles", "wins_rl",
        "wins_opponent", "draws", "raw_win_rate_rl", "mirror_score_rl",
        "team_assignment_gap", "source", "model_dir", "evaluation_dir",
    ]
    _write_csv(
        aggregate / "learning_curve.csv",
        [asdict(row) for row in eval_rows],
        eval_fields,
    )

    training_fields = [
        "algorithm", "seed", "checkpoint", "training_opponent",
        "requested_segment_timesteps", "actual_environment_steps",
        "episodes_segment", "mean_reward_segment", "last_window_reward_segment",
        "mean_episode_length_segment", "mean_loss_segment", "epsilon_end",
        "gradient_updates", "replay_size", "source", "monitor_file",
    ]
    _write_csv(
        aggregate / "training_progress.csv",
        training_rows,
        training_fields,
    )

    grouped: dict[tuple[str, int, str, str], list[EvalRow]] = {}
    for row in eval_rows:
        grouped.setdefault(
            (row.algorithm, row.checkpoint, row.opponent, row.evaluation_kind),
            [],
        ).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (algorithm, checkpoint, opponent, kind), rows in sorted(grouped.items()):
        scores = [row.mirror_score_rl for row in rows]
        aggregate_rows.append({
            "algorithm": algorithm,
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
        aggregate_rows,
        [
            "algorithm", "checkpoint", "opponent", "evaluation_kind",
            "n_seeds", "mean_mirror_score", "sample_sd_mirror_score",
            "min_mirror_score", "max_mirror_score", "total_wins_rl",
            "total_wins_opponent", "total_draws", "total_battles",
        ],
    )

    finals = [row for row in eval_rows if row.evaluation_kind == "final"]
    final_grouped: dict[tuple[str, str], list[EvalRow]] = {}
    for row in finals:
        final_grouped.setdefault((row.algorithm, row.opponent), []).append(row)

    final_summary: list[dict[str, Any]] = []
    for (algorithm, opponent), rows in sorted(final_grouped.items()):
        scores = [row.mirror_score_rl for row in rows]
        final_summary.append({
            "algorithm": algorithm,
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
            "algorithm", "opponent", "n_seeds", "mean_mirror_score",
            "sample_sd_mirror_score", "min_mirror_score", "max_mirror_score",
            "total_wins_rl", "total_wins_opponent", "total_draws",
            "total_battles",
        ],
    )

    by_key = {
        (row.algorithm, row.seed, row.opponent): row
        for row in finals
    }
    seeds = sorted({row.seed for row in finals})
    opponents = sorted({row.opponent for row in finals})
    algorithms = sorted({row.algorithm for row in finals})

    paired: list[dict[str, Any]] = []
    for algorithm in algorithms:
        if algorithm == "maskable_ppo":
            continue
        for opponent in opponents:
            for seed in seeds:
                baseline = by_key.get(("maskable_ppo", seed, opponent))
                candidate = by_key.get((algorithm, seed, opponent))
                if baseline is None or candidate is None:
                    continue
                paired.append({
                    "algorithm": algorithm,
                    "seed": seed,
                    "opponent": opponent,
                    "ppo_score": baseline.mirror_score_rl,
                    "algorithm_score": candidate.mirror_score_rl,
                    "difference_vs_ppo": (
                        candidate.mirror_score_rl - baseline.mirror_score_rl
                    ),
                })

    _write_csv(
        aggregate / "paired_algorithm_differences.csv",
        paired,
        [
            "algorithm", "seed", "opponent", "ppo_score",
            "algorithm_score", "difference_vs_ppo",
        ],
    )

    comparison: list[dict[str, Any]] = []
    for algorithm in algorithms:
        if algorithm == "maskable_ppo":
            continue
        for opponent in opponents:
            rows = [
                row for row in paired
                if row["algorithm"] == algorithm
                and row["opponent"] == opponent
            ]
            if not rows:
                continue
            diffs = [float(row["difference_vs_ppo"]) for row in rows]
            comparison.append({
                "algorithm": algorithm,
                "opponent": opponent,
                "paired_seeds": len(rows),
                "mean_ppo": statistics.fmean(float(row["ppo_score"]) for row in rows),
                "mean_algorithm": statistics.fmean(float(row["algorithm_score"]) for row in rows),
                "mean_difference_vs_ppo": statistics.fmean(diffs),
                "sample_sd_difference": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
            })

    _write_csv(
        aggregate / "algorithm_comparison.csv",
        comparison,
        [
            "algorithm", "opponent", "paired_seeds", "mean_ppo",
            "mean_algorithm", "mean_difference_vs_ppo",
            "sample_sd_difference",
        ],
    )

    figures = aggregate / "figures"
    figures.mkdir(exist_ok=True)
    for opponent in sorted({row.opponent for row in eval_rows}):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for algorithm in sorted({row.algorithm for row in eval_rows}):
            points = [
                row for row in aggregate_rows
                if row["algorithm"] == algorithm
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
                label=algorithm,
            )
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Nominal training timesteps")
        ax.set_ylabel("Mean mirrored score (± seed SD)")
        ax.set_title(f"RL algorithm learning curves vs {opponent}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"learning_curve_vs_{opponent}.png", dpi=160)
        plt.close(fig)


def _study_manifest(
    args: argparse.Namespace,
    stages: list[Stage],
    pool: dict[str, Any],
    dqn_config: DQNConfig,
) -> dict[str, Any]:
    eff_segment = effective_segment_steps(
        args.checkpoint_interval,
        args.ppo_n_steps,
    )
    return {
        "project": "PokeBench30",
        "version": __version__,
        "experiment_type": "rl_algorithm_study",
        "generation": int(args.generation),
        "representation": args.ubr,
        "policy_architecture": "mlp_baseline",
        "algorithms": list(args.algorithms),
        "seeds": [int(v) for v in args.seeds],
        "total_timesteps": int(args.total_timesteps),
        "checkpoint_interval": int(args.checkpoint_interval),
        "effective_segment_environment_steps": int(eff_segment),
        "effective_total_environment_steps": int(
            eff_segment
            * len(
                nominal_checkpoints(
                    args.total_timesteps,
                    args.checkpoint_interval,
                )
            )
        ),
        "checkpoint_repeats_per_leg": int(args.checkpoint_repeats),
        "final_repeats_per_leg": int(args.final_repeats),
        "evaluation_opponents": list(args.eval_opponents),
        "curriculum": [asdict(stage) for stage in stages],
        "ppo": {
            "learning_rate": float(args.ppo_learning_rate),
            "n_steps": int(args.ppo_n_steps),
            "batch_size": int(args.ppo_batch_size),
            "net_arch": [256, 128],
        },
        "dqn_shared": dqn_config.to_dict(),
        "ppo_baseline_dir": (
            str(Path(args.ppo_baseline_dir).resolve())
            if args.ppo_baseline_dir else None
        ),
        "fixed_team_pool": {
            "config_sha256": pool.get("config_sha256"),
            "showdown_commit": pool.get("showdown_commit"),
            "showdown_dirty": pool.get("showdown_dirty"),
        },
        "design_notes": [
            "v0.5.6 holds UBR 1.2, reward, MLP [256,128], curriculum, seeds, effective environment-interaction budget and evaluation fixed.",
            "DQN and Double DQN share all hyperparameters; their only implementation difference is the bootstrap target rule.",
            "Illegal actions are masked both during behavior selection and next-state bootstrap targets.",
            "The DQN replay buffer is retained across the Random->TypeAware curriculum transition.",
            "The PPO baseline can be imported from the completed v0.5.5/v0.5.4 experiment.",
            "Algorithm-specific optimization machinery necessarily differs between PPO and Q-learning; this is an algorithm-level comparison, not a single-hyperparameter ablation.",
        ],
    }


def _same_manifest(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = [
        "generation", "representation", "policy_architecture",
        "algorithms", "seeds", "total_timesteps", "checkpoint_interval",
        "effective_segment_environment_steps",
        "effective_total_environment_steps",
        "checkpoint_repeats_per_leg", "final_repeats_per_leg",
        "evaluation_opponents", "curriculum", "ppo", "dqn_shared",
        "ppo_baseline_dir", "fixed_team_pool",
    ]
    return all(
        _canonical(left.get(key)) == _canonical(right.get(key))
        for key in keys
    )


def run_study(args: argparse.Namespace) -> Path:
    config = load_config(Path(args.config))
    schema = build_feature_schema(args.ubr)
    reward = RewardConfig.from_mapping(config.get("reward"))
    stages = parse_curriculum(config)

    if args.smoke:
        args.seeds = [int(args.seeds[0])]
        args.total_timesteps = 6144
        args.checkpoint_interval = 2048
        args.checkpoint_repeats = 1
        args.final_repeats = 1
        args.ppo_n_steps = 256
        args.ppo_batch_size = 64
        args.ppo_baseline_dir = None
        args.dqn_learning_starts = 256
        args.dqn_replay_capacity = 4096
        args.dqn_target_update_interval = 256
        stages = [
            Stage("random", 2048),
            Stage("type_aware", 4096),
        ]

    validate_curriculum(
        stages,
        total_timesteps=args.total_timesteps,
        checkpoint_interval=args.checkpoint_interval,
    )

    checkpoints = nominal_checkpoints(
        args.total_timesteps,
        args.checkpoint_interval,
    )
    segment_effective = effective_segment_steps(
        args.checkpoint_interval,
        args.ppo_n_steps,
    )
    total_effective = segment_effective * len(checkpoints)

    dqn_config = DQNConfig(
        learning_rate=float(args.dqn_learning_rate),
        gamma=float(args.dqn_gamma),
        batch_size=int(args.dqn_batch_size),
        replay_capacity=int(args.dqn_replay_capacity),
        learning_starts=int(args.dqn_learning_starts),
        train_frequency=int(args.dqn_train_frequency),
        gradient_steps=int(args.dqn_gradient_steps),
        target_update_interval=int(args.dqn_target_update_interval),
        epsilon_start=float(args.dqn_epsilon_start),
        epsilon_final=float(args.dqn_epsilon_final),
        epsilon_fraction=float(args.dqn_epsilon_fraction),
        max_grad_norm=float(args.dqn_max_grad_norm),
        hidden_sizes=(256, 128),
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
        / f"algorithm_gen{args.generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    study_root.mkdir(parents=True, exist_ok=True)

    requested_manifest = _study_manifest(
        args,
        stages,
        pool_manifest,
        dqn_config,
    )
    manifest_path = study_root / "study_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _same_manifest(existing, requested_manifest):
            raise ValueError(
                "The selected --study-dir contains a different algorithm-study "
                "configuration. Use the same arguments or a new directory."
            )
    else:
        manifest_path.write_text(
            json.dumps(requested_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    all_eval_rows: list[EvalRow] = []
    all_training_rows: list[dict[str, Any]] = []
    imported_ppo = False

    if args.ppo_baseline_dir and "maskable_ppo" in args.algorithms:
        imported_eval, imported_training = import_ppo_baseline(
            Path(args.ppo_baseline_dir),
            args=args,
            stages=stages,
            pool_manifest=pool_manifest,
            snapshot_dir=study_root / "imported_maskable_ppo",
        )
        all_eval_rows.extend(imported_eval)
        all_training_rows.extend(imported_training)
        imported_ppo = True
        print(
            f"[IMPORT] MaskablePPO baseline: "
            f"{len(imported_training)} training rows, "
            f"{len(imported_eval)} evaluation rows."
        )

    train_algorithms = [
        algorithm
        for algorithm in args.algorithms
        if not (algorithm == "maskable_ppo" and imported_ppo)
    ]

    print(f"[INFO] Algorithm-study directory: {study_root}")
    print(f"[INFO] Algorithms: {args.algorithms}")
    print(f"[INFO] Algorithms to train now: {train_algorithms}")
    print(f"[INFO] Seeds: {args.seeds}")
    print(f"[INFO] Nominal checkpoints: {checkpoints}")
    print(
        f"[INFO] Effective environment steps/segment: {segment_effective}; "
        f"total: {total_effective}"
    )

    battle_format = f"gen{args.generation}randombattle"

    for algorithm in train_algorithms:
        for seed in args.seeds:
            run_root = study_root / "runs" / algorithm / f"seed_{int(seed)}"
            checkpoint_root = run_root / "checkpoints"
            logs_root = run_root / "training_logs"
            eval_root = run_root / "evaluations"
            tensorboard_root = run_root / "tensorboard"
            resume_root = run_root / "resume"
            for directory in (
                checkpoint_root, logs_root, eval_root,
                tensorboard_root, resume_root,
            ):
                directory.mkdir(parents=True, exist_ok=True)

            current_nominal = _latest_contiguous_checkpoint(
                checkpoint_root,
                algorithm=algorithm,
                checkpoints=checkpoints,
                generation=args.generation,
                representation_version=args.ubr,
                seed=int(seed),
            )

            ppo_model = None
            dqn_trainer = None

            if current_nominal:
                checkpoint_dir = checkpoint_root / f"step_{current_nominal:06d}"
                if algorithm == "maskable_ppo":
                    from sb3_contrib import MaskablePPO
                    ppo_model = MaskablePPO.load(
                        str(checkpoint_dir / "maskable_ppo.zip"),
                        device=args.device,
                    )
                elif current_nominal < args.total_timesteps:
                    marker = resume_root / "resume_metadata.json"
                    trainer_state = resume_root / "trainer_state.pt"
                    replay_state = resume_root / "replay_buffer.npz"
                    if not (marker.exists() and trainer_state.exists() and replay_state.exists()):
                        raise RuntimeError(
                            f"Exact DQN resume state is missing for "
                            f"{algorithm} seed={seed}. Refusing to silently reset replay."
                        )
                    info = json.loads(marker.read_text(encoding="utf-8"))
                    if int(info["nominal_checkpoint"]) != int(current_nominal):
                        raise RuntimeError(
                            "DQN resume replay does not match latest checkpoint."
                        )
                    dqn_trainer = MaskedDQNTrainer.load_training_state(
                        model_path=trainer_state,
                        replay_path=replay_state,
                        device=args.device,
                    )

                print(
                    f"[RESUME] {algorithm} seed={seed}: "
                    f"checkpoint {current_nominal}."
                )

            for checkpoint in checkpoints:
                checkpoint_dir = checkpoint_root / f"step_{checkpoint:06d}"
                previous_nominal = checkpoint - args.checkpoint_interval
                training_opponent = opponent_at(stages, previous_nominal)
                monitor_prefix = (
                    logs_root
                    / f"segment_{previous_nominal:06d}_{checkpoint:06d}_{training_opponent}.csv"
                )

                if checkpoint > current_nominal:
                    # Remove an interrupted monitor segment before replaying it
                    # from the previous exact checkpoint.
                    _remove_partial_monitor(monitor_prefix)

                    env = make_env(
                        battle_format=battle_format,
                        opponent_key=training_opponent,
                        representation_version=args.ubr,
                        reward=reward,
                        log_level=args.log_level,
                        monitor_prefix=monitor_prefix,
                    )

                    print(
                        f"[TRAIN] {algorithm} seed={seed}: nominal "
                        f"{previous_nominal}->{checkpoint} "
                        f"({segment_effective} effective env steps) "
                        f"vs {training_opponent}"
                    )

                    try:
                        if algorithm == "maskable_ppo":
                            if ppo_model is None:
                                ppo_model = _ppo_model(
                                    env=env,
                                    seed=int(seed),
                                    args=args,
                                    config=config,
                                    tensorboard_dir=tensorboard_root,
                                )
                            else:
                                ppo_model.set_env(env)

                            ppo_model.learn(
                                total_timesteps=int(segment_effective),
                                reset_num_timesteps=False,
                                progress_bar=False,
                                tb_log_name="maskable_ppo",
                            )
                            _save_ppo_checkpoint(
                                model=ppo_model,
                                directory=checkpoint_dir,
                                generation=args.generation,
                                representation_version=args.ubr,
                                checkpoint=checkpoint,
                                seed=int(seed),
                                reward=reward,
                                config={
                                    **config,
                                    "algorithm": {
                                        **config["algorithm"],
                                        "n_steps": args.ppo_n_steps,
                                        "batch_size": args.ppo_batch_size,
                                        "learning_rate": args.ppo_learning_rate,
                                    },
                                },
                            )
                            stats = _read_monitor_rewards(monitor_prefix)
                            training_row = {
                                "algorithm": algorithm,
                                "seed": int(seed),
                                "checkpoint": checkpoint,
                                "training_opponent": training_opponent,
                                "requested_segment_timesteps": int(args.checkpoint_interval),
                                "actual_environment_steps": int(ppo_model.num_timesteps),
                                "episodes_segment": stats["episodes"],
                                "mean_reward_segment": stats["mean_reward"],
                                "last_window_reward_segment": stats["last_window_reward"],
                                "mean_episode_length_segment": stats["mean_episode_length"],
                                "mean_loss_segment": "",
                                "epsilon_end": "",
                                "gradient_updates": "",
                                "replay_size": "",
                                "source": "trained_v0.5.6",
                                "monitor_file": str(_monitor_path(monitor_prefix) or monitor_prefix),
                            }

                        else:
                            if dqn_trainer is None:
                                dqn_trainer = MaskedDQNTrainer(
                                    observation_dim=schema.observation_dim,
                                    action_dim=schema.action_dim,
                                    algorithm=ALGORITHM_METADATA_NAMES[algorithm],
                                    config=dqn_config,
                                    total_environment_steps=total_effective,
                                    seed=int(seed),
                                    device=args.device,
                                )
                                print(
                                    f"[DQN] {algorithm}: "
                                    f"params={dqn_trainer.parameter_count():,}, "
                                    f"replay={dqn_config.replay_capacity:,}"
                                )

                            stats = _train_dqn_segment(
                                trainer=dqn_trainer,
                                env=env,
                                steps=segment_effective,
                            )
                            _save_dqn_checkpoint(
                                trainer=dqn_trainer,
                                directory=checkpoint_dir,
                                algorithm_key=algorithm,
                                generation=args.generation,
                                representation_version=args.ubr,
                                checkpoint=checkpoint,
                                seed=int(seed),
                                reward=reward,
                            )
                            (checkpoint_dir / "segment_metrics.json").write_text(
                                json.dumps(
                                    {
                                        "training_opponent": training_opponent,
                                        "requested_segment_timesteps": int(args.checkpoint_interval),
                                        "effective_segment_environment_steps": int(segment_effective),
                                        **stats,
                                    },
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                            # Exact crash-safe state. Only one replay snapshot
                            # is kept per seed, so disk use stays bounded.
                            dqn_trainer.save_resume_state(
                                resume_root,
                                nominal_checkpoint=checkpoint,
                            )
                            if checkpoint == args.total_timesteps:
                                # Final checkpoint itself is enough for future
                                # evaluation resume; the large replay snapshot
                                # is no longer needed.
                                for name in (
                                    "trainer_state.pt",
                                    "replay_buffer.npz",
                                    "resume_metadata.json",
                                ):
                                    candidate = resume_root / name
                                    if candidate.exists():
                                        candidate.unlink()

                            training_row = {
                                "algorithm": algorithm,
                                "seed": int(seed),
                                "checkpoint": checkpoint,
                                "training_opponent": training_opponent,
                                "requested_segment_timesteps": int(args.checkpoint_interval),
                                "actual_environment_steps": int(dqn_trainer.environment_steps),
                                "episodes_segment": stats["episodes"],
                                "mean_reward_segment": stats["mean_reward"],
                                "last_window_reward_segment": stats["last_window_reward"],
                                "mean_episode_length_segment": stats["mean_episode_length"],
                                "mean_loss_segment": stats["mean_loss"],
                                "epsilon_end": stats["epsilon_end"],
                                "gradient_updates": stats["gradient_updates"],
                                "replay_size": stats["replay_size"],
                                "source": "trained_v0.5.6",
                                "monitor_file": str(_monitor_path(monitor_prefix) or monitor_prefix),
                            }
                    finally:
                        env.close()

                    all_training_rows.append(training_row)
                    current_nominal = checkpoint

                else:
                    # Rebuild the aggregate row for already completed training.
                    metadata = load_model_metadata(checkpoint_dir)
                    summary_path = checkpoint_dir / "training_state_summary.json"
                    dqn_summary = (
                        json.loads(summary_path.read_text(encoding="utf-8"))
                        if summary_path.exists() else {}
                    )
                    metrics_path = checkpoint_dir / "segment_metrics.json"
                    stored_metrics = (
                        json.loads(metrics_path.read_text(encoding="utf-8"))
                        if metrics_path.exists() else {}
                    )
                    stats = _read_monitor_rewards(monitor_prefix)
                    all_training_rows.append({
                        "algorithm": algorithm,
                        "seed": int(seed),
                        "checkpoint": checkpoint,
                        "training_opponent": training_opponent,
                        "requested_segment_timesteps": int(args.checkpoint_interval),
                        "actual_environment_steps": (
                            dqn_summary.get("environment_steps", "")
                            if algorithm != "maskable_ppo"
                            else ""
                        ),
                        "episodes_segment": stats["episodes"],
                        "mean_reward_segment": stats["mean_reward"],
                        "last_window_reward_segment": stats["last_window_reward"],
                        "mean_episode_length_segment": stats["mean_episode_length"],
                        "mean_loss_segment": stored_metrics.get("mean_loss", ""),
                        "epsilon_end": stored_metrics.get(
                            "epsilon_end",
                            dqn_summary.get("epsilon", ""),
                        ),
                        "gradient_updates": stored_metrics.get(
                            "gradient_updates",
                            dqn_summary.get("gradient_updates", ""),
                        ),
                        "replay_size": stored_metrics.get(
                            "replay_size",
                            dqn_summary.get("replay_size", ""),
                        ),
                        "source": "trained_v0.5.6",
                        "monitor_file": str(_monitor_path(monitor_prefix) or monitor_prefix),
                    })

                kind = "final" if checkpoint == args.total_timesteps else "diagnostic"
                repeats = (
                    args.final_repeats if kind == "final"
                    else args.checkpoint_repeats
                )
                for opponent in args.eval_opponents:
                    all_eval_rows.append(
                        evaluate_checkpoint(
                            algorithm=algorithm,
                            seed=int(seed),
                            checkpoint=checkpoint,
                            opponent=opponent,
                            kind=kind,
                            repeats=repeats,
                            model_dir=checkpoint_dir,
                            eval_root=eval_root,
                            args=args,
                        )
                    )

                _write_aggregates(
                    study_root,
                    all_eval_rows,
                    all_training_rows,
                )

    _write_aggregates(study_root, all_eval_rows, all_training_rows)

    print("\n=== PokéBench30 v0.5.6 RL algorithm study complete ===")
    print(f"Study: {study_root}")
    print(f"Final summary: {study_root / 'aggregate' / 'final_summary.csv'}")
    print(
        "Algorithm comparison: "
        f"{study_root / 'aggregate' / 'algorithm_comparison.csv'}"
    )
    return study_root


def build_parser() -> argparse.ArgumentParser:
    config = load_config(DEFAULT_CONFIG)
    block = config["algorithm_study"]
    ppo = config["algorithm"]
    dqn = block["dqn"]

    parser = argparse.ArgumentParser(
        description=(
            "PokéBench30 v0.5.6 RL algorithm study: "
            "MaskablePPO vs masked DQN vs masked Double DQN."
        )
    )
    parser.add_argument("--generation", type=int, choices=[1, 4, 9], default=9)
    parser.add_argument(
        "--ubr",
        choices=["ubr-1.1", "ubr-1.2"],
        default=block.get("representation", "ubr-1.2"),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=list(ALGORITHM_KEYS),
        default=block.get("algorithms", list(ALGORITHM_KEYS)),
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
        "--ppo-baseline-dir",
        default=None,
        help=(
            "Completed v0.5.5 policy-study or v0.5.4 curriculum-study "
            "directory used to import the MaskablePPO baseline."
        ),
    )

    parser.add_argument(
        "--ppo-learning-rate",
        type=float,
        default=float(ppo.get("learning_rate", 3e-4)),
    )
    parser.add_argument(
        "--ppo-n-steps",
        type=int,
        default=int(ppo.get("n_steps", 2048)),
    )
    parser.add_argument(
        "--ppo-batch-size",
        type=int,
        default=int(ppo.get("batch_size", 128)),
    )

    parser.add_argument("--dqn-learning-rate", type=float, default=float(dqn["learning_rate"]))
    parser.add_argument("--dqn-gamma", type=float, default=float(dqn["gamma"]))
    parser.add_argument("--dqn-batch-size", type=int, default=int(dqn["batch_size"]))
    parser.add_argument("--dqn-replay-capacity", type=int, default=int(dqn["replay_capacity"]))
    parser.add_argument("--dqn-learning-starts", type=int, default=int(dqn["learning_starts"]))
    parser.add_argument("--dqn-train-frequency", type=int, default=int(dqn["train_frequency"]))
    parser.add_argument("--dqn-gradient-steps", type=int, default=int(dqn["gradient_steps"]))
    parser.add_argument("--dqn-target-update-interval", type=int, default=int(dqn["target_update_interval"]))
    parser.add_argument("--dqn-epsilon-start", type=float, default=float(dqn["epsilon_start"]))
    parser.add_argument("--dqn-epsilon-final", type=float, default=float(dqn["epsilon_final"]))
    parser.add_argument("--dqn-epsilon-fraction", type=float, default=float(dqn["epsilon_fraction"]))
    parser.add_argument("--dqn-max-grad-norm", type=float, default=float(dqn["max_grad_norm"]))

    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pool-root", default=str(DEFAULT_POOL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--study-dir", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One-seed 6144-step end-to-end test of all three algorithms.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.algorithms:
        raise ValueError("At least one algorithm is required.")
    if len(set(args.algorithms)) != len(args.algorithms):
        raise ValueError("Algorithms must be unique.")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be nonempty and unique.")
    if args.checkpoint_repeats <= 0 or args.final_repeats <= 0:
        raise ValueError("Evaluation repeats must be > 0.")

    run_study(args)


if __name__ == "__main__":
    main()
