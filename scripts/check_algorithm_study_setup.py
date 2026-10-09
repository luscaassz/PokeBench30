from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pokebench30.representation.schema import build_feature_schema
from pokebench30.rl.algorithm_study import (
    effective_segment_steps,
    import_ppo_baseline,
    load_config,
    nominal_checkpoints,
    parse_curriculum,
)
from pokebench30.rl.masked_dqn import DQNConfig
from pokebench30.teams.fixed_pool import (
    DEFAULT_POOL_ROOT,
    load_pool_manifest,
    verify_pool,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check PokéBench30 v0.5.6 algorithm-study prerequisites."
    )
    parser.add_argument("--ppo-baseline-dir", default=None)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "config" / "rl_baseline.yaml")
    block = config["algorithm_study"]
    ppo = config["algorithm"]
    dqn_raw = block["dqn"]
    schema = build_feature_schema(block["representation"])

    print(
        f"[OK] Representation: {schema.version} "
        f"obs={schema.observation_dim}, actions={schema.action_dim}, "
        f"sha={schema.sha256}"
    )
    print("[OK] Policy architecture fixed: MLP [256, 128]")
    print(
        "[OK] Algorithms: "
        + ", ".join(block["algorithms"])
    )

    segment_effective = effective_segment_steps(
        block["checkpoint_interval"],
        ppo["n_steps"],
    )
    cps = nominal_checkpoints(
        block["total_timesteps"],
        block["checkpoint_interval"],
    )
    print(
        f"[OK] Effective environment budget: "
        f"{segment_effective} per nominal 50k segment; "
        f"{segment_effective * len(cps)} total"
    )

    dqn_config = DQNConfig(
        learning_rate=float(dqn_raw["learning_rate"]),
        gamma=float(dqn_raw["gamma"]),
        batch_size=int(dqn_raw["batch_size"]),
        replay_capacity=int(dqn_raw["replay_capacity"]),
        learning_starts=int(dqn_raw["learning_starts"]),
        train_frequency=int(dqn_raw["train_frequency"]),
        gradient_steps=int(dqn_raw["gradient_steps"]),
        target_update_interval=int(dqn_raw["target_update_interval"]),
        epsilon_start=float(dqn_raw["epsilon_start"]),
        epsilon_final=float(dqn_raw["epsilon_final"]),
        epsilon_fraction=float(dqn_raw["epsilon_fraction"]),
        max_grad_norm=float(dqn_raw["max_grad_norm"]),
        hidden_sizes=(256, 128),
    )
    print(f"[OK] Shared DQN/DDQN config: {dqn_config.to_dict()}")

    problems = verify_pool(DEFAULT_POOL_ROOT)
    if problems:
        print("[FAIL] Fixed-team pool is not ready:")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nCopy the validated pool from v0.5.5, for example:\n"
            '  Copy-Item "..\\PokeBench30_v0.5.5\\teams\\fixed" '
            '".\\teams\\fixed" -Recurse -Force'
        )
        raise SystemExit(1)

    pool_manifest = load_pool_manifest(DEFAULT_POOL_ROOT)
    print(f"[OK] Fixed-team pool: {DEFAULT_POOL_ROOT}")

    if args.ppo_baseline_dir:
        class Args:
            generation = 9
            ubr = block["representation"]
            seeds = block["seeds"]
            total_timesteps = block["total_timesteps"]
            checkpoint_interval = block["checkpoint_interval"]
            checkpoint_repeats = block["checkpoint_repeats_per_leg"]
            final_repeats = block["final_repeats_per_leg"]
            eval_opponents = block["evaluation_opponents"]
            ppo_learning_rate = ppo["learning_rate"]
            ppo_n_steps = ppo["n_steps"]
            ppo_batch_size = ppo["batch_size"]

        eval_rows, training_rows = import_ppo_baseline(
            Path(args.ppo_baseline_dir),
            args=Args,
            stages=parse_curriculum(config),
            pool_manifest=pool_manifest,
            snapshot_dir=PROJECT_ROOT / ".algorithm_setup_check_tmp",
        )
        tmp = PROJECT_ROOT / ".algorithm_setup_check_tmp"
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)
        print(
            f"[OK] PPO baseline import: "
            f"{len(training_rows)} training rows, "
            f"{len(eval_rows)} evaluation rows."
        )
    else:
        print(
            "[INFO] No PPO baseline supplied. "
            "MaskablePPO will be trained again."
        )

    print("[INFO] Fixed curriculum: Random 100k -> TypeAware 200k")
    print("[INFO] Full study: 5 seeds × 300k nominal")
    print("[OK] v0.5.6 algorithm-study setup is ready.")


if __name__ == "__main__":
    main()
