from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pokebench30.representation.schema import build_feature_schema
from pokebench30.rl.crossgen_study import (
    expected_actual_final,
    load_config,
    nominal_checkpoints,
    rollout_transition_block,
    validate_fixed_pool,
)
from pokebench30.rl.wang_protocol import (
    WANG_INITIAL_LR,
    WangLearningRateSchedule,
)
from pokebench30.teams.fixed_pool import DEFAULT_POOL_ROOT


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check PokéBench30 v0.6.0 cross-generation prerequisites."
    )
    parser.add_argument(
        "--pool-root",
        default=str(DEFAULT_POOL_ROOT),
    )
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "config" / "crossgen_benchmark.yaml")
    study = config["study"]
    ppo = config["ppo"]

    schema = build_feature_schema("ubr-1.2")
    print(
        f"[OK] UBR: {schema.version}, obs={schema.observation_dim}, "
        f"actions={schema.action_dim}, sha={schema.sha256}"
    )

    if schema.observation_dim != 518 or schema.action_dim != 10:
        raise SystemExit("[FAIL] v0.6.0 requires UBR 1.2 with 518x10 interface.")

    print("[OK] Frozen generations:", study["generations"])
    print("[OK] Frozen seeds:", study["seeds"])
    print("[OK] Training regime: shared-policy two-player PettingZoo self-play")
    print("[OK] Reward: sparse terminal {-1, 0, +1}")

    schedule = WangLearningRateSchedule()
    print(
        f"[OK] Wang LR: initial={WANG_INITIAL_LR:.10g}, "
        f"mid={schedule(0.5):.10g}, final={schedule(0.0):.10g}"
    )

    print(
        "[OK] PPO: "
        f"epochs={ppo['n_epochs']}, gamma={ppo['gamma']}, "
        f"gae={ppo['gae_lambda']}, clip={ppo['clip_range']}, "
        f"vf_clip={ppo['clip_range_vf']}, ent={ppo['ent_coef']}, "
        f"vf_coef={ppo['vf_coef']}, max_grad={ppo['max_grad_norm']}"
    )
    print(
        "[OK] PettingZoo rollout: "
        f"n_steps={ppo['n_steps']} × 2 players × "
        f"{ppo.get('parallel_env_copies', 1)} env copy = "
        f"{rollout_transition_block(ppo)} transitions/update; "
        f"batch={ppo['batch_size']}"
    )
    print(
        "[OK] Network: shared "
        f"{ppo['trunk_layers']}x{ppo['hidden_dim']} trunk + "
        f"{ppo['head_layers']}x{ppo['hidden_dim']} actor/critic heads"
    )

    checkpoints = nominal_checkpoints(
        int(study["total_timesteps"]),
        int(study["checkpoint_interval"]),
    )
    actual_final = expected_actual_final(
        int(study["total_timesteps"]),
        rollout_transition_block(ppo),
    )
    print(f"[OK] Checkpoints: {checkpoints}")
    print(
        f"[OK] Budget/model: nominal={study['total_timesteps']}, "
        f"expected actual={actual_final}"
    )

    pool = validate_fixed_pool(
        Path(args.pool_root).resolve(),
        generations=[int(v) for v in study["generations"]],
        expected_pairs=int(study["fixed_team_pairs_per_generation"]),
    )
    print(
        "[OK] Fixed-team pool: "
        f"{study['fixed_team_pairs_per_generation']} pairs/gen, "
        f"config_sha={pool.get('config_sha256')}"
    )
    print(
        "[OK] Showdown provenance: "
        f"commit={pool.get('showdown_commit')}, "
        f"dirty={pool.get('showdown_dirty')}"
    )

    print(
        "[INFO] Installed: "
        f"poke-env={_version('poke-env')}, "
        f"stable-baselines3={_version('stable-baselines3')}, "
        f"sb3-contrib={_version('sb3-contrib')}, "
        f"torch={_version('torch')}"
    )

    print("[OK] PokéBench30 v0.6.0 setup is ready.")


if __name__ == "__main__":
    main()
