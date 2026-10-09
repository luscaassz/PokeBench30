from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pokebench30.representation.schema import build_feature_schema
from pokebench30.rl.architecture_study import (
    import_v054_baseline,
    load_config,
    parse_fixed_curriculum,
)
from pokebench30.rl.policy_specs import (
    DEFAULT_POLICY_ARCHITECTURES,
    get_policy_spec,
)
from pokebench30.teams.fixed_pool import (
    DEFAULT_POOL_ROOT,
    load_pool_manifest,
    verify_pool,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check PokéBench30 v0.5.5 policy-study prerequisites."
    )
    parser.add_argument("--baseline-curriculum-dir", default=None)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "config" / "rl_baseline.yaml")
    block = config["policy_study"]
    algo = config["algorithm"]
    schema = build_feature_schema(block["representation"])

    print(
        f"[OK] Representation: {schema.version} "
        f"obs={schema.observation_dim}, actions={schema.action_dim}, "
        f"sha={schema.sha256}"
    )

    for name in block.get("architectures", DEFAULT_POLICY_ARCHITECTURES):
        spec = get_policy_spec(name)
        print(
            f"[OK] {name}: family={spec.family}, "
            f"history={spec.history_len}, "
            f"model_input_dim={spec.model_input_dim(schema.observation_dim)}, "
            f"net_arch={list(spec.net_arch)}"
        )

    problems = verify_pool(DEFAULT_POOL_ROOT)
    if problems:
        print("[FAIL] Fixed-team pool is not ready:")
        for problem in problems:
            print(f"  - {problem}")
        print(
            '\nPrepare it with:\n'
            '  python prepare_fixed_teams.py --showdown-dir '
            '"C:\\path\\to\\pokemon-showdown" --force'
        )
        raise SystemExit(1)

    pool_manifest = load_pool_manifest(DEFAULT_POOL_ROOT)
    print(f"[OK] Fixed-team pool: {DEFAULT_POOL_ROOT}")

    if args.baseline_curriculum_dir:
        class Args:
            generation = 9
            ubr = block["representation"]
            seeds = block["seeds"]
            total_timesteps = block["total_timesteps"]
            checkpoint_interval = block["checkpoint_interval"]
            checkpoint_repeats = block["checkpoint_repeats_per_leg"]
            final_repeats = block["final_repeats_per_leg"]
            eval_opponents = block["evaluation_opponents"]
            n_steps = algo["n_steps"]
            batch_size = algo["batch_size"]
            learning_rate = algo["learning_rate"]

        eval_rows, training_rows = import_v054_baseline(
            Path(args.baseline_curriculum_dir),
            args=Args,
            stages=parse_fixed_curriculum(config),
            pool_manifest=pool_manifest,
            snapshot_dir=None,
        )
        print(
            f"[OK] v0.5.4 baseline import: "
            f"{len(training_rows)} training rows, "
            f"{len(eval_rows)} evaluation rows."
        )
    else:
        print(
            "[INFO] No v0.5.4 baseline supplied. "
            "mlp_baseline will be trained from scratch."
        )

    print("[INFO] Fixed curriculum: Random 100k -> TypeAware 200k")
    print("[INFO] Default total budget: 300k per architecture/seed")
    print("[OK] v0.5.5 policy-study setup is ready.")


if __name__ == "__main__":
    main()
