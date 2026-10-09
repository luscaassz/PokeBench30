from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pokebench30.representation.schema import build_feature_schema
from pokebench30.rl.budget_study import (
    import_baseline_experiment,
    load_rl_config,
)
from pokebench30.teams.fixed_pool import (
    DEFAULT_POOL_ROOT,
    load_pool_manifest,
    verify_pool,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check PokéBench30 v0.5.3 budget-study prerequisites."
    )
    parser.add_argument("--baseline-experiment-dir", default=None)
    args = parser.parse_args()

    config = load_rl_config(PROJECT_ROOT / "config" / "rl_baseline.yaml")
    block = config["budget_study"]
    algo = config["algorithm"]

    for version in block["representations"]:
        schema = build_feature_schema(version)
        print(
            f"[OK] {version}: obs={schema.observation_dim}, "
            f"actions={schema.action_dim}, sha={schema.sha256}"
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

    if args.baseline_experiment_dir:
        training, evaluations = import_baseline_experiment(
            Path(args.baseline_experiment_dir),
            baseline_budget=int(block["baseline_budget"]),
            generation=9,
            ubrrs=list(block["representations"]),
            seeds=[int(x) for x in block["seeds"]],
            training_opponent=block["training_opponent"],
            eval_opponents=list(block["evaluation_opponents"]),
            repeats=int(block["repeats_per_leg"]),
            n_steps=int(algo["n_steps"]),
            batch_size=int(algo["batch_size"]),
            learning_rate=float(algo["learning_rate"]),
            current_pool_manifest=pool_manifest,
            snapshot_dir=None,
        )
        print(
            f"[OK] Imported-baseline compatibility: "
            f"{len(training)} model rows, {len(evaluations)} evaluation rows"
        )
    else:
        print(
            "[INFO] No baseline experiment supplied. The full study will "
            "train every requested budget from scratch."
        )

    budgets = block["budgets"]
    seeds = block["seeds"]
    reps = block["representations"]
    opponents = block["evaluation_opponents"]

    print(f"[INFO] Default budgets: {budgets}")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Representations: {reps}")
    print(f"[INFO] Evaluation opponents: {opponents}")
    print("[OK] Budget-study prerequisites are ready.")


if __name__ == "__main__":
    main()
