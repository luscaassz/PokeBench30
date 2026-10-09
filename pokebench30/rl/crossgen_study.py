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

import yaml

from pokebench30 import __version__
from pokebench30.representation.schema import build_feature_schema
from pokebench30.rl.model_io import (
    ModelMetadata,
    load_model_metadata,
    validate_model_metadata,
)
from pokebench30.teams.fixed_pool import (
    DEFAULT_POOL_ROOT,
    config_hash,
    load_pool_manifest,
    records_by_key,
    verify_pool,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "crossgen_benchmark.yaml"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "crossgen_results"


@dataclass(frozen=True)
class EvalRow:
    generation: int
    seed: int
    checkpoint: int
    actual_model_timesteps: int
    opponent: str
    evaluation_kind: str
    repeats_per_leg: int
    team_pairs: int
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
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def nominal_checkpoints(total_timesteps: int, interval: int) -> list[int]:
    total_timesteps = int(total_timesteps)
    interval = int(interval)
    if total_timesteps <= 0 or interval <= 0:
        raise ValueError("Timesteps and checkpoint interval must be > 0.")
    if total_timesteps % interval != 0:
        raise ValueError(
            "total_timesteps must be divisible by checkpoint_interval."
        )
    return list(range(interval, total_timesteps + 1, interval))


def actual_rollout_checkpoint(
    nominal_step: int,
    rollout_transitions: int,
) -> int:
    """First complete PPO rollout boundary at/after a nominal checkpoint."""
    return int(
        math.ceil(int(nominal_step) / int(rollout_transitions))
        * int(rollout_transitions)
    )


def rollout_transition_block(ppo_config: dict[str, Any]) -> int:
    # PettingZoo SinglesEnv exposes two agents. SuperSuit can optionally
    # duplicate the whole environment, so the SB3 VecEnv size is:
    # 2 players × num_env_copies.
    return (
        int(ppo_config["n_steps"])
        * 2
        * int(ppo_config.get("parallel_env_copies", 1))
    )


def expected_actual_final(
    total_timesteps: int,
    rollout_transitions: int,
) -> int:
    return actual_rollout_checkpoint(
        total_timesteps,
        rollout_transitions,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _canonical(value: Any) -> Any:
    return json.loads(
        json.dumps(value, sort_keys=True, ensure_ascii=False)
    )


def validate_fixed_pool(
    pool_root: Path,
    *,
    generations: list[int],
    expected_pairs: int,
) -> dict[str, Any]:
    problems = verify_pool(pool_root)
    if problems:
        raise RuntimeError(
            "Fixed-team pool failed integrity verification:\n- "
            + "\n- ".join(problems)
        )

    manifest = load_pool_manifest(pool_root)
    records = records_by_key(manifest)

    extra: list[str] = []
    current_config_hash = config_hash()
    if manifest.get("config_sha256") != current_config_hash:
        extra.append(
            "Fixed-team manifest was generated from a different "
            "config/fixed_teams.yaml. Regenerate the v0.6 pool with --force."
        )
    for generation in generations:
        pair_ids = sorted({
            pair_id
            for gen, pair_id, side in records
            if gen == int(generation)
        })
        if len(pair_ids) != int(expected_pairs):
            extra.append(
                f"Gen {generation}: expected {expected_pairs} team pairs, "
                f"found {len(pair_ids)} ({pair_ids})."
            )
        for pair_id in pair_ids:
            for side in ("A", "B"):
                if (int(generation), pair_id, side) not in records:
                    extra.append(
                        f"Gen {generation} {pair_id}: missing side {side}."
                    )

    if extra:
        raise RuntimeError(
            "Cross-generation fixed pool is incompatible:\n- "
            + "\n- ".join(extra)
        )

    return manifest


def _model_metadata(
    *,
    generation: int,
    seed: int,
    nominal_checkpoint: int,
    actual_timesteps: int,
    representation_version: str,
    config: dict[str, Any],
) -> ModelMetadata:
    from pokebench30.rl.wang_style import (
        WANG_INITIAL_LR,
        WangPPOConfig,
    )

    schema = build_feature_schema(representation_version)
    ppo = WangPPOConfig(
        n_epochs=int(config["ppo"]["n_epochs"]),
        gamma=float(config["ppo"]["gamma"]),
        gae_lambda=float(config["ppo"]["gae_lambda"]),
        clip_range=float(config["ppo"]["clip_range"]),
        clip_range_vf=float(config["ppo"]["clip_range_vf"]),
        ent_coef=float(config["ppo"]["ent_coef"]),
        vf_coef=float(config["ppo"]["vf_coef"]),
        max_grad_norm=float(config["ppo"]["max_grad_norm"]),
        n_steps=int(config["ppo"]["n_steps"]),
        batch_size=int(config["ppo"]["batch_size"]),
        hidden_dim=int(config["ppo"]["hidden_dim"]),
        trunk_layers=int(config["ppo"]["trunk_layers"]),
        head_layers=int(config["ppo"]["head_layers"]),
    )

    return ModelMetadata(
        project_version=__version__,
        algorithm="PPO",
        training_generation=int(generation),
        training_battle_format=f"gen{int(generation)}randombattle",
        training_opponent="shared_policy_self_play",
        total_timesteps=int(nominal_checkpoint),
        seed=int(seed),
        observation_schema_version=schema.version,
        observation_schema_sha256=schema.sha256,
        observation_dim=schema.observation_dim,
        action_dim=schema.action_dim,
        reward={
            "win_value": 1.0,
            "loss_value": -1.0,
            "other_value": 0.0,
        },
        library_versions={
            "poke-env": _version("poke-env"),
            "gymnasium": _version("gymnasium"),
            "stable-baselines3": _version("stable-baselines3"),
            "sb3-contrib": _version("sb3-contrib"),
            "torch": _version("torch"),
            "numpy": _version("numpy"),
        },
        notes=[
            "PokéBench30 v0.6 cross-generation benchmark.",
            "Training uses Wang-inspired shared-policy self-play.",
            "Both player trajectories are added to the PPO rollout buffer through the two-agent PettingZoo environment.",
            "Reward is sparse terminal: +1 win, -1 loss, 0 otherwise.",
            "Learning-rate annealing follows 10^-4.23 / (8*x + 1)^1.5.",
            "PPO hyperparameters follow Wang (MIT, 2024) where they map cleanly.",
            "PokéBench30 UBR 1.2 replaces Wang's generation-specific 3725-dimensional representation.",
            "The policy uses a shared 3x256 ReLU trunk and separate 2x256 actor/critic heads.",
            "The common-core 10-action benchmark excludes generation-specific gimmicks such as Terastallization.",
            f"Nominal checkpoint={nominal_checkpoint}; actual model timesteps={actual_timesteps}.",
        ],
        policy_architecture="wang_style_mlp",
        policy_config={
            "family": "wang_style_mlp",
            "shared_trunk_layers": int(ppo.trunk_layers),
            "shared_hidden_dim": int(ppo.hidden_dim),
            "actor_head_layers": int(ppo.head_layers),
            "critic_head_layers": int(ppo.head_layers),
            "activation": "ReLU",
        },
        model_input_dim=schema.observation_dim,
        algorithm_config={
            **ppo.to_dict(),
            "learning_rate_initial": WANG_INITIAL_LR,
            "training_regime": "shared_policy_pettingzoo_self_play",
            "reward_profile": "wang_sparse_terminal",
            "parallel_env_copies": int(
                config["ppo"].get("parallel_env_copies", 1)
            ),
            "player_perspectives_per_env": 2,
        },
    )


def _write_model_metadata(
    checkpoint_dir: Path,
    **kwargs: Any,
) -> None:
    metadata = _model_metadata(**kwargs)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _latest_checkpoint(
    checkpoint_root: Path,
    *,
    checkpoints: list[int],
    generation: int,
    seed: int,
    representation_version: str,
) -> int:
    latest = 0
    for checkpoint in checkpoints:
        directory = checkpoint_root / f"step_{checkpoint:06d}"
        model_file = directory / "ppo.zip"
        metadata_file = directory / "metadata.json"
        if not (model_file.exists() and metadata_file.exists()):
            break

        metadata = load_model_metadata(directory)
        validate_model_metadata(
            metadata,
            generation=generation,
            expected_representation_version=representation_version,
        )
        if int(metadata.seed) != int(seed):
            raise ValueError(
                f"Checkpoint seed mismatch at {directory}."
            )
        if int(metadata.total_timesteps) != int(checkpoint):
            raise ValueError(
                f"Checkpoint nominal step mismatch at {directory}."
            )
        if metadata.training_opponent != "shared_policy_self_play":
            raise ValueError(
                f"Checkpoint {directory} is not a v0.6 self-play model."
            )
        latest = checkpoint
    return latest



def _patch_supersuit_seed_compat(vec_env):
    """Bridge SB3's VecEnv.seed() call to Gymnasium reset(seed=...).

    SuperSuit's current ConcatVecEnv follows the Gymnasium API and exposes
    ``reset(seed=...)`` but not a legacy ``seed()`` method. Stable-Baselines3's
    VecEnv wrapper still calls ``venv.seed(seed)`` during algorithm setup.

    We install a tiny compatibility shim on the underlying SuperSuit vector
    environment. The seed is stored and consumed by the *next* reset so model
    construction does not start and discard an extra Pokémon battle.
    """
    base = getattr(vec_env, "venv", None)
    if base is None:
        return vec_env
    if hasattr(base, "seed"):
        return vec_env
    if not hasattr(base, "reset"):
        return vec_env

    original_reset = base.reset
    pending = {"seed": None}

    def seed(seed_value=None):
        pending["seed"] = seed_value
        count = int(getattr(base, "num_envs", getattr(vec_env, "num_envs", 1)))
        if seed_value is None:
            return [None] * count
        start = int(seed_value)
        return [start + index for index in range(count)]

    def reset(*args, seed=None, options=None, **kwargs):
        if seed is None and pending["seed"] is not None:
            seed = pending["seed"]
            pending["seed"] = None
        return original_reset(
            *args,
            seed=seed,
            options=options,
            **kwargs,
        )

    base.seed = seed
    base.reset = reset
    return vec_env

def _make_training_env(
    *,
    generation: int,
    representation_version: str,
    log_level: int,
    num_env_copies: int,
):
    try:
        import supersuit as ss
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "SuperSuit is required for two-player self-play. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc

    from pokebench30.rl.wang_style import SparseTerminalSelfPlayEnv

    battle_format = f"gen{int(generation)}randombattle"

    pettingzoo_env = SparseTerminalSelfPlayEnv(
        battle_format=battle_format,
        representation_version=representation_version,
        log_level=int(log_level),
        open_timeout=None,
    )

    vec_env = ss.pettingzoo_env_to_vec_env_v1(pettingzoo_env)
    vec_env = ss.concat_vec_envs_v1(
        vec_env,
        num_vec_envs=int(num_env_copies),
        num_cpus=int(num_env_copies),
        base_class="stable_baselines3",
    )
    vec_env = _patch_supersuit_seed_compat(vec_env)
    return vec_env


def _make_model(
    *,
    env,
    seed: int,
    config: dict[str, Any],
    device: str,
    tensorboard_dir: Path,
):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "stable-baselines3 is required. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc

    from pokebench30.rl.wang_protocol import (
        WangLearningRateSchedule,
        WangPPOConfig,
    )
    from pokebench30.rl.wang_style import (
        WangMaskedActorCriticPolicy,
        wang_policy_kwargs,
    )

    raw = config["ppo"]
    ppo = WangPPOConfig(
        n_epochs=int(raw["n_epochs"]),
        gamma=float(raw["gamma"]),
        gae_lambda=float(raw["gae_lambda"]),
        clip_range=float(raw["clip_range"]),
        clip_range_vf=float(raw["clip_range_vf"]),
        ent_coef=float(raw["ent_coef"]),
        vf_coef=float(raw["vf_coef"]),
        max_grad_norm=float(raw["max_grad_norm"]),
        n_steps=int(raw["n_steps"]),
        batch_size=int(raw["batch_size"]),
        hidden_dim=int(raw["hidden_dim"]),
        trunk_layers=int(raw["trunk_layers"]),
        head_layers=int(raw["head_layers"]),
    )

    return PPO(
        WangMaskedActorCriticPolicy,
        env,
        learning_rate=WangLearningRateSchedule(),
        n_steps=int(ppo.n_steps),
        batch_size=int(ppo.batch_size),
        n_epochs=int(ppo.n_epochs),
        gamma=float(ppo.gamma),
        gae_lambda=float(ppo.gae_lambda),
        clip_range=float(ppo.clip_range),
        clip_range_vf=float(ppo.clip_range_vf),
        ent_coef=float(ppo.ent_coef),
        vf_coef=float(ppo.vf_coef),
        max_grad_norm=float(ppo.max_grad_norm),
        policy_kwargs=wang_policy_kwargs(ppo),
        seed=int(seed),
        device=device,
        tensorboard_log=str(tensorboard_dir),
        verbose=1,
    )


def _checkpoint_callback(
    *,
    checkpoint_root: Path,
    checkpoints: list[int],
    completed_nominal: int,
    generation: int,
    seed: int,
    representation_version: str,
    config: dict[str, Any],
):
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required.") from exc

    class CrossGenCheckpointCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.pending = [
                int(step)
                for step in checkpoints
                if int(step) > int(completed_nominal)
            ]

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            while (
                self.pending
                and int(self.model.num_timesteps) >= self.pending[0]
            ):
                nominal = self.pending.pop(0)
                directory = checkpoint_root / f"step_{nominal:06d}"
                directory.mkdir(parents=True, exist_ok=True)
                self.model.save(str(directory / "ppo"))

                ep_infos = list(getattr(self.model, "ep_info_buffer", []) or [])
                rewards = [
                    float(info["r"])
                    for info in ep_infos
                    if "r" in info
                ]
                lengths = [
                    float(info["l"])
                    for info in ep_infos
                    if "l" in info
                ]

                schedule = self.model.lr_schedule
                current_lr = float(
                    schedule(
                        float(
                            getattr(
                                self.model,
                                "_current_progress_remaining",
                                0.0,
                            )
                        )
                    )
                )

                rollout_rewards = []
                rollout_buffer = getattr(self.model, "rollout_buffer", None)
                if rollout_buffer is not None:
                    try:
                        import numpy as np
                        rollout_rewards = np.asarray(
                            rollout_buffer.rewards,
                            dtype=float,
                        ).reshape(-1).tolist()
                    except Exception:
                        rollout_rewards = []

                metrics = {
                    "generation": int(generation),
                    "seed": int(seed),
                    "nominal_checkpoint": int(nominal),
                    "actual_model_timesteps": int(self.model.num_timesteps),
                    "learning_rate": current_lr,
                    "last100_episode_reward_mean": (
                        statistics.fmean(rewards)
                        if rewards else None
                    ),
                    "last100_episode_length_mean": (
                        statistics.fmean(lengths)
                        if lengths else None
                    ),
                    "ep_info_count": len(ep_infos),
                    "rollout_transition_count": len(rollout_rewards),
                    "rollout_terminal_positive": sum(
                        value > 0.0 for value in rollout_rewards
                    ),
                    "rollout_terminal_negative": sum(
                        value < 0.0 for value in rollout_rewards
                    ),
                    "rollout_zero_rewards": sum(
                        value == 0.0 for value in rollout_rewards
                    ),
                }
                (directory / "training_metrics.json").write_text(
                    json.dumps(
                        metrics,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                _write_model_metadata(
                    directory,
                    generation=generation,
                    seed=seed,
                    nominal_checkpoint=nominal,
                    actual_timesteps=int(self.model.num_timesteps),
                    representation_version=representation_version,
                    config=config,
                )

                print(
                    f"[CHECKPOINT] Gen {generation} seed={seed}: "
                    f"nominal {nominal} at actual "
                    f"{int(self.model.num_timesteps)}"
                )

    return CrossGenCheckpointCallback()


def _load_eval_summary(
    *,
    generation: int,
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
            f"Expected one controlled_summary row at {eval_dir}; "
            f"got {len(rows)}."
        )
    row = rows[0]

    metadata = load_model_metadata(model_dir)
    return EvalRow(
        generation=int(generation),
        seed=int(seed),
        checkpoint=int(checkpoint),
        actual_model_timesteps=int(
            json.loads(
                (model_dir / "training_metrics.json").read_text(
                    encoding="utf-8"
                )
            )["actual_model_timesteps"]
        ),
        opponent=opponent,
        evaluation_kind=kind,
        repeats_per_leg=int(repeats),
        team_pairs=int(row["team_pairs"]),
        battles=int(row["battles"]),
        mirror_cycles=int(row["mirror_cycles"]),
        wins_rl=int(row["wins_agent_1"]),
        wins_opponent=int(row["wins_agent_2"]),
        draws=int(row["draws"]),
        raw_win_rate_rl=float(row["raw_win_rate_agent_1"]),
        mirror_score_rl=float(
            row["normalized_mirror_score_agent_1"]
        ),
        team_assignment_gap=float(
            row["agent_1_team_assignment_gap"]
        ),
        model_dir=str(model_dir),
        evaluation_dir=str(eval_dir),
    )


def _evaluate_checkpoint(
    *,
    generation: int,
    seed: int,
    checkpoint: int,
    model_dir: Path,
    opponent: str,
    kind: str,
    repeats: int,
    args: argparse.Namespace,
    eval_root: Path,
) -> EvalRow:
    run_name = f"step_{checkpoint:06d}_vs_{opponent}_{kind}"
    eval_dir = eval_root / run_name
    summary = eval_dir / "controlled_summary.csv"

    if not summary.exists():
        if eval_dir.exists():
            shutil.rmtree(eval_dir)

        command = [
            sys.executable,
            str(PROJECT_ROOT / "run_rl_controlled.py"),
            "--model", str(model_dir),
            "--generation", str(generation),
            "--opponent", opponent,
            "--repeats", str(repeats),
            "--concurrency", str(args.concurrency),
            "--pool-root", str(Path(args.pool_root).resolve()),
            "--output-root", str(eval_root),
            "--run-name", run_name,
            "--no-action-log",
            "--no-plots",
        ]
        evaluation_pairs = getattr(args, "evaluation_pairs", None)
        if evaluation_pairs:
            command.extend(["--pairs", *evaluation_pairs])

        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
        )
    else:
        print(f"[RESUME] Existing evaluation accepted: {eval_dir}")

    return _load_eval_summary(
        generation=generation,
        seed=seed,
        checkpoint=checkpoint,
        opponent=opponent,
        kind=kind,
        repeats=repeats,
        model_dir=model_dir,
        eval_dir=eval_dir,
    )


def _collect_pair_rows(eval_rows: list[EvalRow]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in eval_rows:
        if row.evaluation_kind != "final":
            continue
        path = Path(row.evaluation_dir) / "pair_summary.csv"
        if not path.exists():
            continue
        for pair in _read_csv(path):
            output.append({
                "generation": row.generation,
                "seed": row.seed,
                "checkpoint": row.checkpoint,
                "opponent": row.opponent,
                "pair_id": pair["pair_id"],
                "battles": int(pair["battles"]),
                "mirror_cycles": int(pair["mirror_cycles"]),
                "mirror_score_rl": float(
                    pair["normalized_mirror_score_agent_1"]
                ),
                "wins_rl": int(pair["wins_agent_1"]),
                "wins_opponent": int(pair["wins_agent_2"]),
                "draws": int(pair["draws"]),
                "team_assignment_gap": float(
                    pair["agent_1_team_assignment_gap"]
                ),
            })
    return output


def _training_rows(
    study_root: Path,
    *,
    generations: list[int],
    seeds: list[int],
    checkpoints: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for generation in generations:
        for seed in seeds:
            checkpoint_root = (
                study_root
                / "runs"
                / f"gen{generation}"
                / f"seed_{seed}"
                / "checkpoints"
            )
            for checkpoint in checkpoints:
                path = (
                    checkpoint_root
                    / f"step_{checkpoint:06d}"
                    / "training_metrics.json"
                )
                if not path.exists():
                    continue
                raw = json.loads(path.read_text(encoding="utf-8"))
                rows.append(raw)
    return rows


def _write_aggregates(
    study_root: Path,
    eval_rows: list[EvalRow],
    training_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> None:
    aggregate = study_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)

    eval_fields = list(asdict(eval_rows[0]).keys()) if eval_rows else [
        "generation", "seed", "checkpoint", "actual_model_timesteps",
        "opponent", "evaluation_kind", "repeats_per_leg", "team_pairs",
        "battles", "mirror_cycles", "wins_rl", "wins_opponent", "draws",
        "raw_win_rate_rl", "mirror_score_rl", "team_assignment_gap",
        "model_dir", "evaluation_dir",
    ]
    _write_csv(
        aggregate / "learning_curve.csv",
        [asdict(row) for row in eval_rows],
        eval_fields,
    )

    _write_csv(
        aggregate / "training_progress.csv",
        training_rows,
        [
            "generation", "seed", "nominal_checkpoint",
            "actual_model_timesteps", "learning_rate",
            "last100_episode_reward_mean",
            "last100_episode_length_mean", "ep_info_count",
            "rollout_transition_count",
            "rollout_terminal_positive",
            "rollout_terminal_negative",
            "rollout_zero_rewards",
        ],
    )

    if pair_rows:
        _write_csv(
            aggregate / "pair_level_final.csv",
            pair_rows,
            [
                "generation", "seed", "checkpoint", "opponent", "pair_id",
                "battles", "mirror_cycles", "mirror_score_rl",
                "wins_rl", "wins_opponent", "draws",
                "team_assignment_gap",
            ],
        )

    grouped: dict[tuple[int, int, str, str], list[EvalRow]] = {}
    for row in eval_rows:
        grouped.setdefault(
            (
                row.generation,
                row.checkpoint,
                row.opponent,
                row.evaluation_kind,
            ),
            [],
        ).append(row)

    aggregate_curve: list[dict[str, Any]] = []
    for (generation, checkpoint, opponent, kind), rows in sorted(
        grouped.items()
    ):
        scores = [row.mirror_score_rl for row in rows]
        aggregate_curve.append({
            "generation": generation,
            "checkpoint": checkpoint,
            "opponent": opponent,
            "evaluation_kind": kind,
            "n_seeds": len(rows),
            "mean_mirror_score": statistics.fmean(scores),
            "sample_sd_mirror_score": (
                statistics.stdev(scores)
                if len(scores) > 1 else 0.0
            ),
            "min_mirror_score": min(scores),
            "max_mirror_score": max(scores),
            "total_battles": sum(row.battles for row in rows),
        })

    _write_csv(
        aggregate / "aggregate_learning_curve.csv",
        aggregate_curve,
        [
            "generation", "checkpoint", "opponent", "evaluation_kind",
            "n_seeds", "mean_mirror_score", "sample_sd_mirror_score",
            "min_mirror_score", "max_mirror_score", "total_battles",
        ],
    )

    finals = [
        row for row in eval_rows
        if row.evaluation_kind == "final"
    ]
    final_groups: dict[tuple[int, str], list[EvalRow]] = {}
    for row in finals:
        final_groups.setdefault(
            (row.generation, row.opponent),
            [],
        ).append(row)

    final_summary: list[dict[str, Any]] = []
    for (generation, opponent), rows in sorted(final_groups.items()):
        scores = [row.mirror_score_rl for row in rows]
        final_summary.append({
            "generation": generation,
            "opponent": opponent,
            "n_seeds": len(rows),
            "mean_mirror_score": statistics.fmean(scores),
            "sample_sd_mirror_score": (
                statistics.stdev(scores)
                if len(scores) > 1 else 0.0
            ),
            "min_mirror_score": min(scores),
            "max_mirror_score": max(scores),
            "total_wins_rl": sum(row.wins_rl for row in rows),
            "total_wins_opponent": sum(row.wins_opponent for row in rows),
            "total_draws": sum(row.draws for row in rows),
            "total_battles": sum(row.battles for row in rows),
        })

    _write_csv(
        aggregate / "generation_summary.csv",
        final_summary,
        [
            "generation", "opponent", "n_seeds", "mean_mirror_score",
            "sample_sd_mirror_score", "min_mirror_score",
            "max_mirror_score", "total_wins_rl",
            "total_wins_opponent", "total_draws", "total_battles",
        ],
    )

    by_key = {
        (row.generation, row.seed, row.opponent): row
        for row in finals
    }
    generations = sorted({row.generation for row in finals})
    seeds = sorted({row.seed for row in finals})
    opponents = sorted({row.opponent for row in finals})

    pairwise: list[dict[str, Any]] = []
    for left_index in range(len(generations)):
        for right_index in range(left_index + 1, len(generations)):
            left = generations[left_index]
            right = generations[right_index]
            for opponent in opponents:
                for seed in seeds:
                    row_left = by_key.get((left, seed, opponent))
                    row_right = by_key.get((right, seed, opponent))
                    if row_left is None or row_right is None:
                        continue
                    pairwise.append({
                        "generation_left": left,
                        "generation_right": right,
                        "seed": seed,
                        "opponent": opponent,
                        "score_left": row_left.mirror_score_rl,
                        "score_right": row_right.mirror_score_rl,
                        "difference_right_minus_left": (
                            row_right.mirror_score_rl
                            - row_left.mirror_score_rl
                        ),
                    })

    _write_csv(
        aggregate / "generation_pairwise_differences.csv",
        pairwise,
        [
            "generation_left", "generation_right", "seed", "opponent",
            "score_left", "score_right",
            "difference_right_minus_left",
        ],
    )

    comparisons: list[dict[str, Any]] = []
    compare_groups: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in pairwise:
        compare_groups.setdefault(
            (
                int(row["generation_left"]),
                int(row["generation_right"]),
                str(row["opponent"]),
            ),
            [],
        ).append(row)

    for (left, right, opponent), rows in sorted(compare_groups.items()):
        diffs = [
            float(row["difference_right_minus_left"])
            for row in rows
        ]
        comparisons.append({
            "generation_left": left,
            "generation_right": right,
            "opponent": opponent,
            "paired_seeds": len(rows),
            "mean_left": statistics.fmean(
                float(row["score_left"]) for row in rows
            ),
            "mean_right": statistics.fmean(
                float(row["score_right"]) for row in rows
            ),
            "mean_difference_right_minus_left": statistics.fmean(diffs),
            "sample_sd_difference": (
                statistics.stdev(diffs)
                if len(diffs) > 1 else 0.0
            ),
        })

    _write_csv(
        aggregate / "generation_comparison.csv",
        comparisons,
        [
            "generation_left", "generation_right", "opponent",
            "paired_seeds", "mean_left", "mean_right",
            "mean_difference_right_minus_left",
            "sample_sd_difference",
        ],
    )

    # Diagnostic TypeAware AUC by generation/seed.
    diagnostics = [
        row for row in eval_rows
        if row.evaluation_kind == "diagnostic"
        and row.opponent == "type_aware"
    ]
    auc_rows: list[dict[str, Any]] = []
    diag_groups: dict[tuple[int, int], list[EvalRow]] = {}
    for row in diagnostics:
        diag_groups.setdefault(
            (row.generation, row.seed),
            [],
        ).append(row)

    for (generation, seed), rows in sorted(diag_groups.items()):
        rows.sort(key=lambda row: row.checkpoint)
        if len(rows) < 2:
            continue

        xs = [float(row.checkpoint) for row in rows]
        ys = [float(row.mirror_score_rl) for row in rows]
        area = 0.0
        for index in range(1, len(xs)):
            dx = xs[index] - xs[index - 1]
            area += dx * (ys[index] + ys[index - 1]) / 2.0
        normalized = area / (xs[-1] - xs[0])
        auc_rows.append({
            "generation": generation,
            "seed": seed,
            "opponent": "type_aware",
            "first_checkpoint": int(xs[0]),
            "last_checkpoint": int(xs[-1]),
            "normalized_auc": normalized,
        })

    _write_csv(
        aggregate / "learning_efficiency_typeaware.csv",
        auc_rows,
        [
            "generation", "seed", "opponent",
            "first_checkpoint", "last_checkpoint", "normalized_auc",
        ],
    )

    # Figures are supplemental; raw CSVs remain the source of truth.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figures = aggregate / "figures"
    figures.mkdir(exist_ok=True)

    # Learning curve against the common diagnostic opponent.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for generation in sorted({
        row.generation
        for row in eval_rows
    }):
        points = [
            row for row in aggregate_curve
            if int(row["generation"]) == int(generation)
            and row["opponent"] == "type_aware"
        ]
        points.sort(key=lambda row: int(row["checkpoint"]))
        if not points:
            continue
        ax.errorbar(
            [int(row["checkpoint"]) for row in points],
            [float(row["mean_mirror_score"]) for row in points],
            yerr=[
                float(row["sample_sd_mirror_score"])
                for row in points
            ],
            marker="o",
            capsize=3,
            label=f"Gen {generation}",
        )
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Nominal training timesteps")
    ax.set_ylabel("Mean mirrored score vs TypeAware (± seed SD)")
    ax.set_title("PokéBench30 cross-generation learning curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures / "crossgen_learning_curve_vs_typeaware.png",
        dpi=160,
    )
    plt.close(fig)

    for opponent in opponents:
        rows = [
            row for row in final_summary
            if row["opponent"] == opponent
        ]
        if not rows:
            continue
        rows.sort(key=lambda row: int(row["generation"]))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(
            [int(row["generation"]) for row in rows],
            [float(row["mean_mirror_score"]) for row in rows],
            yerr=[
                float(row["sample_sd_mirror_score"])
                for row in rows
            ],
            marker="o",
            capsize=4,
        )
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_xticks(
            [int(row["generation"]) for row in rows]
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Final mean mirrored score (± seed SD)")
        ax.set_title(f"Final cross-generation score vs {opponent}")
        fig.tight_layout()
        fig.savefig(
            figures / f"final_score_vs_{opponent}.png",
            dpi=160,
        )
        plt.close(fig)


def _study_manifest(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    pool_manifest: dict[str, Any],
) -> dict[str, Any]:
    ppo = config["ppo"]
    checkpoints = nominal_checkpoints(
        args.total_timesteps,
        args.checkpoint_interval,
    )

    return {
        "project": "PokeBench30",
        "version": __version__,
        "experiment_type": "cross_generation_benchmark",
        "generations": [int(value) for value in args.generations],
        "seeds": [int(value) for value in args.seeds],
        "representation": args.ubr,
        "action_space": {
            "name": "universal_core_v1",
            "dimension": 10,
            "generation_specific_gimmicks": "excluded",
        },
        "training": {
            "format": "generation-specific random battles",
            "regime": "shared_policy_pettingzoo_self_play",
            "rollout_trajectory_source": "both_players",
            "reward": {
                "win": 1.0,
                "loss": -1.0,
                "other": 0.0,
            },
            "nominal_total_timesteps": int(args.total_timesteps),
            "rollout_transition_block": rollout_transition_block(ppo),
            "expected_actual_final_timesteps": expected_actual_final(
                args.total_timesteps,
                rollout_transition_block(ppo),
            ),
        },
        "ppo": dict(ppo),
        "learning_rate_schedule": "10^-4.23 / (8*x + 1)^1.5",
        "checkpoints": checkpoints,
        "checkpoint_interval": int(args.checkpoint_interval),
        "diagnostic_opponents": list(args.diagnostic_opponents),
        "diagnostic_repeats_per_leg": int(args.diagnostic_repeats),
        "final_opponents": list(args.final_opponents),
        "final_repeats_per_leg": int(args.final_repeats),
        "fixed_team_pairs_per_generation": int(args.expected_pairs),
        "fixed_team_pool": {
            "config_sha256": pool_manifest.get("config_sha256"),
            "showdown_commit": pool_manifest.get("showdown_commit"),
            "showdown_dirty": pool_manifest.get("showdown_dirty"),
        },
        "design_notes": [
            "The v0.5.x ablations are treated as pilot/engineering validation rather than the central scientific contribution.",
            "v0.6 freezes one literature-grounded PPO configuration and compares Gen I, IV and IX.",
            "Each generation is trained independently from scratch with identical hyperparameters and seeds.",
            "Both player perspectives share the same PPO policy and both trajectories enter the rollout buffer.",
            "Primary evaluation is same-generation; zero-shot transfer is not part of v0.6.0.",
            "Ten fixed mirrored team pairs per generation are used to reduce dependence on only three pair units.",
            "Final evaluation uses 3 repeats per mirror leg: 10 pairs x 2 legs x 3 = 60 battles per seed/opponent.",
            "Gen IX Terastallization is intentionally excluded from the common-core action space to preserve identical action semantics across generations.",
        ],
    }


def _same_manifest(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    keys = [
        "generations",
        "seeds",
        "representation",
        "action_space",
        "training",
        "ppo",
        "learning_rate_schedule",
        "checkpoints",
        "checkpoint_interval",
        "diagnostic_opponents",
        "diagnostic_repeats_per_leg",
        "final_opponents",
        "final_repeats_per_leg",
        "fixed_team_pairs_per_generation",
        "fixed_team_pool",
    ]
    return all(
        _canonical(left.get(key)) == _canonical(right.get(key))
        for key in keys
    )


def _train_one(
    *,
    study_root: Path,
    generation: int,
    seed: int,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required.") from exc

    # Ensure custom serialized classes are importable before loading.
    from pokebench30.rl import wang_style as _wang_style  # noqa: F401

    run_root = (
        study_root
        / "runs"
        / f"gen{generation}"
        / f"seed_{seed}"
    )
    checkpoint_root = run_root / "checkpoints"
    tensorboard_root = run_root / "tensorboard"
    training_logs_root = run_root / "training_logs"
    eval_root = run_root / "evaluations"

    for directory in (
        checkpoint_root,
        tensorboard_root,
        training_logs_root,
        eval_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoints = nominal_checkpoints(
        args.total_timesteps,
        args.checkpoint_interval,
    )
    completed_nominal = _latest_checkpoint(
        checkpoint_root,
        checkpoints=checkpoints,
        generation=generation,
        seed=seed,
        representation_version=args.ubr,
    )

    if completed_nominal == args.total_timesteps:
        print(
            f"[RESUME] Gen {generation} seed={seed}: "
            "training already complete."
        )
        return

    start_actual = 0
    if completed_nominal:
        completed_dir = (
            checkpoint_root
            / f"step_{completed_nominal:06d}"
        )
        metrics = json.loads(
            (completed_dir / "training_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        start_actual = int(metrics["actual_model_timesteps"])

    env = _make_training_env(
        generation=generation,
        representation_version=args.ubr,
        log_level=args.log_level,
        num_env_copies=int(config["ppo"].get("parallel_env_copies", 1)),
    )

    try:
        if completed_nominal:
            model_path = (
                checkpoint_root
                / f"step_{completed_nominal:06d}"
                / "ppo.zip"
            )
            model = PPO.load(
                str(model_path),
                env=env,
                device=args.device,
            )
            print(
                f"[RESUME] Gen {generation} seed={seed}: "
                f"nominal checkpoint {completed_nominal}, "
                f"actual steps={model.num_timesteps}."
            )
        else:
            model = _make_model(
                env=env,
                seed=seed,
                config=config,
                device=args.device,
                tensorboard_dir=tensorboard_root,
            )
            print(
                f"[TRAIN] Gen {generation} seed={seed}: "
                "new Wang-style two-player self-play model."
            )

        callback = _checkpoint_callback(
            checkpoint_root=checkpoint_root,
            checkpoints=checkpoints,
            completed_nominal=completed_nominal,
            generation=generation,
            seed=seed,
            representation_version=args.ubr,
            config=config,
        )

        remaining = max(
            0,
            int(args.total_timesteps) - int(model.num_timesteps),
        )
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                callback=callback,
                reset_num_timesteps=(completed_nominal == 0),
                progress_bar=False,
                tb_log_name=f"gen{generation}_seed{seed}",
            )
    finally:
        env.close()

    final_dir = (
        checkpoint_root
        / f"step_{args.total_timesteps:06d}"
    )
    if not (final_dir / "ppo.zip").exists():
        raise RuntimeError(
            f"Final checkpoint was not written for Gen {generation} "
            f"seed={seed}."
        )


def run_study(args: argparse.Namespace) -> Path:
    config = load_config(Path(args.config))

    if args.smoke:
        args.generations = [1, 4, 9]
        args.seeds = [int(args.seeds[0])]
        args.total_timesteps = 8192
        args.checkpoint_interval = 4096
        args.diagnostic_repeats = 1
        args.final_repeats = 1
        args.evaluation_pairs = ["pair01", "pair02", "pair03"]

        # Smoke must remain fast while still exercising Wang-style PPO.
        config = json.loads(json.dumps(config))
        config["ppo"]["n_steps"] = 512
        config["ppo"]["batch_size"] = 256

    if not hasattr(args, "evaluation_pairs"):
        args.evaluation_pairs = None

    schema = build_feature_schema(args.ubr)
    if schema.observation_dim != 518:
        raise ValueError(
            "PokéBench30 v0.6.0 is frozen to UBR 1.2 (518 features)."
        )

    checkpoints = nominal_checkpoints(
        args.total_timesteps,
        args.checkpoint_interval,
    )

    pool_root = Path(args.pool_root).resolve()
    pool_manifest = validate_fixed_pool(
        pool_root,
        generations=[int(value) for value in args.generations],
        expected_pairs=args.expected_pairs,
    )

    study_root = (
        Path(args.study_dir).resolve()
        if args.study_dir
        else Path(args.output_root).resolve()
        / f"crossgen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    study_root.mkdir(parents=True, exist_ok=True)

    requested_manifest = _study_manifest(
        args=args,
        config=config,
        pool_manifest=pool_manifest,
    )
    manifest_path = study_root / "study_manifest.json"

    if manifest_path.exists():
        existing = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if not _same_manifest(existing, requested_manifest):
            raise ValueError(
                "The selected --study-dir contains a different "
                "cross-generation benchmark configuration."
            )
    else:
        manifest_path.write_text(
            json.dumps(
                requested_manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"[INFO] Study directory: {study_root}")
    print(f"[INFO] Generations: {args.generations}")
    print(f"[INFO] Seeds: {args.seeds}")
    print(
        f"[INFO] Nominal budget/model: {args.total_timesteps}; "
        f"expected actual final: "
        f"{expected_actual_final(args.total_timesteps, rollout_transition_block(config['ppo']))}"
    )
    print(f"[INFO] Checkpoints: {checkpoints}")
    print(
        f"[INFO] Fixed evaluation: {args.expected_pairs} mirrored "
        "team pairs per generation."
    )

    # Training phase.
    for generation in args.generations:
        for seed in args.seeds:
            _train_one(
                study_root=study_root,
                generation=int(generation),
                seed=int(seed),
                args=args,
                config=config,
            )

    # Evaluation phase. Keeping evaluation after training avoids interrupting
    # the self-play training dynamics and learning-rate schedule.
    eval_rows: list[EvalRow] = []

    for generation in args.generations:
        for seed in args.seeds:
            run_root = (
                study_root
                / "runs"
                / f"gen{int(generation)}"
                / f"seed_{int(seed)}"
            )
            checkpoint_root = run_root / "checkpoints"
            eval_root = run_root / "evaluations"
            eval_root.mkdir(parents=True, exist_ok=True)

            for checkpoint in checkpoints:
                model_dir = (
                    checkpoint_root
                    / f"step_{checkpoint:06d}"
                )

                if checkpoint == args.total_timesteps:
                    opponents = list(args.final_opponents)
                    kind = "final"
                    repeats = int(args.final_repeats)
                else:
                    opponents = list(args.diagnostic_opponents)
                    kind = "diagnostic"
                    repeats = int(args.diagnostic_repeats)

                for opponent in opponents:
                    eval_rows.append(
                        _evaluate_checkpoint(
                            generation=int(generation),
                            seed=int(seed),
                            checkpoint=int(checkpoint),
                            model_dir=model_dir,
                            opponent=opponent,
                            kind=kind,
                            repeats=repeats,
                            args=args,
                            eval_root=eval_root,
                        )
                    )

                    training_rows = _training_rows(
                        study_root,
                        generations=[
                            int(value)
                            for value in args.generations
                        ],
                        seeds=[int(value) for value in args.seeds],
                        checkpoints=checkpoints,
                    )
                    pair_rows = _collect_pair_rows(eval_rows)
                    _write_aggregates(
                        study_root,
                        eval_rows,
                        training_rows,
                        pair_rows,
                    )

    training_rows = _training_rows(
        study_root,
        generations=[int(value) for value in args.generations],
        seeds=[int(value) for value in args.seeds],
        checkpoints=checkpoints,
    )
    pair_rows = _collect_pair_rows(eval_rows)
    _write_aggregates(
        study_root,
        eval_rows,
        training_rows,
        pair_rows,
    )

    print("\n=== PokéBench30 v0.6.0 cross-generation benchmark complete ===")
    print(f"Study: {study_root}")
    print(
        "Generation summary: "
        f"{study_root / 'aggregate' / 'generation_summary.csv'}"
    )
    print(
        "Generation comparison: "
        f"{study_root / 'aggregate' / 'generation_comparison.csv'}"
    )
    print(
        "Pair-level final data: "
        f"{study_root / 'aggregate' / 'pair_level_final.csv'}"
    )
    return study_root


def build_parser() -> argparse.ArgumentParser:
    config = load_config(DEFAULT_CONFIG)
    study = config["study"]

    parser = argparse.ArgumentParser(
        description=(
            "PokéBench30 v0.6.0: literature-grounded Wang-style PPO "
            "cross-generation benchmark for Gen I, IV and IX."
        )
    )
    parser.add_argument(
        "--generations",
        nargs="+",
        type=int,
        choices=[1, 4, 9],
        default=study.get("generations", [1, 4, 9]),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=study.get(
            "seeds",
            [42, 123, 456, 789, 2026],
        ),
    )
    parser.add_argument(
        "--ubr",
        choices=["ubr-1.2"],
        default="ubr-1.2",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=int(study.get("total_timesteps", 900000)),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=int(study.get("checkpoint_interval", 150000)),
    )
    parser.add_argument(
        "--diagnostic-opponents",
        nargs="+",
        default=study.get(
            "diagnostic_opponents",
            ["type_aware"],
        ),
    )
    parser.add_argument(
        "--diagnostic-repeats",
        type=int,
        default=int(
            study.get("diagnostic_repeats_per_leg", 1)
        ),
    )
    parser.add_argument(
        "--final-opponents",
        nargs="+",
        default=study.get(
            "final_opponents",
            [
                "random",
                "max_damage",
                "type_aware",
                "switch_aware",
            ],
        ),
    )
    parser.add_argument(
        "--final-repeats",
        type=int,
        default=int(
            study.get("final_repeats_per_leg", 3)
        ),
    )
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=int(
            study.get("fixed_team_pairs_per_generation", 10)
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pool-root", default=str(DEFAULT_POOL_ROOT))
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_RESULTS_ROOT),
    )
    parser.add_argument(
        "--study-dir",
        default=None,
        help="Resume an existing v0.6 cross-generation study.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a one-seed, 8192-step end-to-end test across "
            "Gen I, IV and IX."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.generations:
        raise ValueError("At least one generation is required.")
    if len(set(args.generations)) != len(args.generations):
        raise ValueError("Generations must be unique.")
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")
    if args.diagnostic_repeats <= 0 or args.final_repeats <= 0:
        raise ValueError("Evaluation repeats must be > 0.")
    if args.expected_pairs <= 0:
        raise ValueError("--expected-pairs must be > 0.")

    run_study(args)


if __name__ == "__main__":
    main()
