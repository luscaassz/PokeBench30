from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pokebench30 import __version__
from pokebench30.representation.schema import build_feature_schema
from pokebench30.rl.curriculum_study import load_config, parse_regimes, validate_regimes
from pokebench30.teams.fixed_pool import DEFAULT_POOL_ROOT, load_pool_manifest, verify_pool


def main() -> None:
    config = load_config(ROOT / "config" / "rl_baseline.yaml")
    block = config["curriculum_study"]
    regimes = parse_regimes(config)

    validate_regimes(
        regimes,
        total_timesteps=int(block["total_timesteps"]),
        checkpoint_interval=int(block["checkpoint_interval"]),
    )

    schema = build_feature_schema(block["representation"])
    print(f"[OK] PokéBench30 version: {__version__}")
    print(
        f"[OK] Representation: {schema.version} | "
        f"obs={schema.observation_dim} | actions={schema.action_dim}"
    )
    print(f"[OK] Seeds: {block['seeds']}")
    print(f"[OK] Total timesteps/regime: {block['total_timesteps']}")
    print(f"[OK] Checkpoint interval: {block['checkpoint_interval']}")

    for name, stages in regimes.items():
        description = " -> ".join(
            f"{stage.opponent}:{stage.timesteps}" for stage in stages
        )
        print(f"[OK] Regime {name}: {description}")

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

    manifest = load_pool_manifest(DEFAULT_POOL_ROOT)
    print(f"[OK] Fixed-team pool: {DEFAULT_POOL_ROOT}")
    print(f"[OK] Showdown commit: {manifest.get('showdown_commit')}")
    print("[OK] Curriculum-study setup is ready.")


if __name__ == "__main__":
    main()
