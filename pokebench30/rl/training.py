from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import yaml

from pokebench30 import __version__
from pokebench30.agents.max_damage_agent import MaxDamageAgent
from pokebench30.agents.random_agent import RandomAgent
from pokebench30.agents.switch_aware_agent import SwitchAwareAgent
from pokebench30.agents.type_aware_agent import TypeAwareAgent
from pokebench30.representation.schema import build_feature_schema
from .model_io import ModelMetadata
from .reward import RewardConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "rl_baseline.yaml"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


OPPONENTS = {
    "random": RandomAgent,
    "max_damage": MaxDamageAgent,
    "type_aware": TypeAwareAgent,
    "switch_aware": SwitchAwareAgent,
}


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def load_rl_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_opponent(
    key: str,
    *,
    battle_format: str,
):
    if key not in OPPONENTS:
        raise ValueError(
            f"Unknown opponent {key!r}. Available: {sorted(OPPONENTS)}"
        )

    return OPPONENTS[key](
        battle_format=battle_format,
        start_listening=False,
        log_actions=False,
        log_universal_state=False,
    )


def train(args: argparse.Namespace) -> Path:
    try:
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from poke_env.environment import SingleAgentWrapper
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise RuntimeError(
            "RL dependencies are not installed. Run:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    from .env import UniversalRLSinglesEnv
    from .mask_adapter import SB3MaskAdapter

    config = load_rl_config(Path(args.config))
    algo = config["algorithm"]
    reward = RewardConfig.from_mapping(config.get("reward"))

    generation = int(args.generation)
    battle_format = f"gen{generation}randombattle"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root).resolve() / (
        f"maskable_ppo_gen{generation}_{run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "tensorboard").mkdir(exist_ok=True)

    base_env = UniversalRLSinglesEnv(
        battle_format=battle_format,
        reward_config=reward,
        log_level=args.log_level,
        open_timeout=None,
    )
    opponent = make_opponent(
        args.opponent,
        battle_format=battle_format,
    )
    single_env = SingleAgentWrapper(base_env, opponent)
    monitored_env = Monitor(
        single_env,
        filename=str(run_dir / "monitor.csv"),
    )
    # Keep the mask adapter outermost so sb3-contrib can call
    # env.action_masks() directly without relying on wrapper attribute lookup.
    env = SB3MaskAdapter(monitored_env)

    policy_kwargs = {
        "net_arch": list(algo.get("policy_net_arch", [256, 128])),
    }

    model = MaskablePPO(
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
        seed=int(args.seed),
        device=args.device,
        tensorboard_log=str(run_dir / "tensorboard"),
        verbose=1,
    )

    checkpoint = CheckpointCallback(
        save_freq=max(1, int(args.checkpoint_freq)),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="maskable_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    try:
        model.learn(
            total_timesteps=int(args.timesteps),
            callback=checkpoint,
            progress_bar=False,
        )
        model_path = run_dir / "maskable_ppo"
        model.save(str(model_path))
    finally:
        env.close()

    schema = build_feature_schema()

    metadata = ModelMetadata(
        project_version=__version__,
        algorithm="MaskablePPO",
        training_generation=generation,
        training_battle_format=battle_format,
        training_opponent=args.opponent,
        total_timesteps=int(args.timesteps),
        seed=int(args.seed),
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
            "Training uses random battles for team diversity.",
            "The policy receives only UBR 1.1 observation features.",
            "Illegal actions are excluded by MaskablePPO action masks.",
            "Generation-specific gimmicks are intentionally outside the v0.5 action space.",
            "Cross-generation evaluation is reserved for PokéBench30 v0.6.",
        ],
    )

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (run_dir / "training_config.json").write_text(
        json.dumps(
            {
                "generation": generation,
                "battle_format": battle_format,
                "opponent": args.opponent,
                "timesteps": int(args.timesteps),
                "seed": int(args.seed),
                "device": args.device,
                "learning_rate": float(args.learning_rate),
                "n_steps": int(args.n_steps),
                "batch_size": int(args.batch_size),
                "checkpoint_freq": int(args.checkpoint_freq),
                "policy_kwargs": policy_kwargs,
                "algorithm_config": algo,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== PokéBench30 v0.5 training complete ===")
    print(f"Model: {run_dir / 'maskable_ppo.zip'}")
    print(f"Metadata: {run_dir / 'metadata.json'}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    config = load_rl_config(DEFAULT_CONFIG)
    defaults = config["training"]
    algo = config["algorithm"]

    parser = argparse.ArgumentParser(
        description="Train the PokéBench30 v0.5 MaskablePPO baseline."
    )
    parser.add_argument(
        "--generation",
        type=int,
        choices=[1, 4, 9],
        default=int(defaults.get("default_generation", 9)),
    )
    parser.add_argument(
        "--opponent",
        choices=sorted(OPPONENTS),
        default=defaults.get("default_opponent", "type_aware"),
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=int(defaults.get("default_timesteps", 10000)),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(defaults.get("default_seed", 42)),
    )
    parser.add_argument(
        "--device",
        default=defaults.get("default_device", "auto"),
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
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_MODELS_DIR),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.timesteps <= 0:
        raise ValueError("--timesteps must be greater than zero.")
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be greater than zero.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")

    train(args)


if __name__ == "__main__":
    main()
