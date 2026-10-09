from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pokebench30.representation.schema import build_feature_schema
from pokebench30.rl.transfer_study import load_config, validate_source_study
from pokebench30.teams.fixed_pool import DEFAULT_POOL_ROOT, load_pool_manifest, verify_pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate PokéBench30 v0.6.1 zero-shot transfer + telemetry setup."
    )
    parser.add_argument("--source-study-dir", required=True)
    parser.add_argument("--pool-root", default=str(DEFAULT_POOL_ROOT))
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "config" / "transfer_telemetry.yaml")
    study = config["study"]
    schema = build_feature_schema("ubr-1.2")

    print(
        f"[OK] UBR: {schema.version}, obs={schema.observation_dim}, "
        f"actions={schema.action_dim}, sha={schema.sha256}"
    )
    print(f"[OK] Generations: {study['generations']}")
    print(f"[OK] Seeds: {study['seeds']}")
    print(f"[OK] Source checkpoint: {study['source_checkpoint']}")
    print(f"[OK] Primary opponents: {study['primary_opponents']}")
    print(f"[INFO] Supplementary opponents: {study['supplementary_opponents']}")
    print(
        f"[OK] Evaluation geometry: {study['team_pairs_per_generation']} pairs × "
        f"2 mirror legs × {study['repeats_per_leg']} repeats = "
        f"{study['team_pairs_per_generation'] * 2 * study['repeats_per_leg']} "
        "battles/seed/opponent/cell"
    )

    pool_root = Path(args.pool_root).resolve()
    problems = verify_pool(pool_root)
    if problems:
        print("[FAIL] Fixed-team pool:")
        for problem in problems:
            print("  -", problem)
        raise SystemExit(1)
    pool = load_pool_manifest(pool_root)
    print(f"[OK] Fixed pool: {pool_root}")
    print(f"[OK] Pool config SHA: {pool.get('config_sha256')}")
    print(f"[OK] Showdown commit: {pool.get('showdown_commit')}")

    manifest = validate_source_study(
        Path(args.source_study_dir),
        config=config,
        pool_manifest=pool,
    )
    print(f"[OK] Source v0.6 study version: {manifest.get('version')}")
    print("[OK] 15 final source models found and metadata-compatible.")

    primary_battles = (
        3 * 3 * len(study["seeds"]) * len(study["primary_opponents"])
        * study["team_pairs_per_generation"] * 2 * study["repeats_per_leg"]
    )
    all_battles = (
        3 * 3 * len(study["seeds"])
        * (len(study["primary_opponents"]) + len(study["supplementary_opponents"]))
        * study["team_pairs_per_generation"] * 2 * study["repeats_per_leg"]
    )
    print(f"[INFO] Default primary study: {primary_battles} controlled battles.")
    print(f"[INFO] With --all-opponents: {all_battles} controlled battles.")
    print("[OK] v0.6.1 setup is ready.")


if __name__ == "__main__":
    main()
