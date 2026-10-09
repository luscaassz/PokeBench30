from __future__ import annotations

import argparse
from pathlib import Path

from pokebench30.teams.fixed_pool import (
    DEFAULT_CONFIG,
    DEFAULT_POOL_ROOT,
    prepare_fixed_pool,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate and freeze deterministic Pokémon Showdown teams for "
            "PokéBench30 v0.3."
        )
    )
    parser.add_argument(
        "--showdown-dir",
        required=True,
        help="Path to the cloned pokemon-showdown repository.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
    )
    parser.add_argument(
        "--pool-root",
        default=str(DEFAULT_POOL_ROOT),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the pool even if a manifest already exists.",
    )
    args = parser.parse_args()

    manifest = prepare_fixed_pool(
        showdown_dir=Path(args.showdown_dir),
        config_path=Path(args.config),
        pool_root=Path(args.pool_root),
        force=args.force,
    )

    print("\nFixed-team pool ready.")
    print(f"Teams: {len(manifest['records'])}")
    print(f"Showdown commit: {manifest.get('showdown_commit')}")
    print(f"Showdown dirty: {manifest.get('showdown_dirty')}")
    print(f"Manifest: {Path(args.pool_root) / 'manifest.json'}")


if __name__ == "__main__":
    main()
