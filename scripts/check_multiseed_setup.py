from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pokebench30.representation.schema import build_feature_schema
from pokebench30.teams.fixed_pool import DEFAULT_POOL_ROOT, verify_pool


def main() -> None:
    for version in ("ubr-1.1", "ubr-1.2"):
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
            '  python prepare_fixed_teams.py --showdown-dir "C:\\path\\to\\pokemon-showdown" --force'
        )
        raise SystemExit(1)

    print(f"[OK] Fixed-team pool: {DEFAULT_POOL_ROOT}")
    print("[OK] Multi-seed experiment prerequisites are ready.")


if __name__ == "__main__":
    main()
