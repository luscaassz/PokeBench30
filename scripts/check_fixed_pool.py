from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pokebench30.teams.fixed_pool import (
    DEFAULT_POOL_ROOT,
    load_pool_manifest,
    verify_pool,
)


def main():
    try:
        manifest = load_pool_manifest(
            DEFAULT_POOL_ROOT
        )
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(1)

    problems = verify_pool(DEFAULT_POOL_ROOT)

    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        raise SystemExit(1)

    print("[OK] Fixed-team manifest found")
    print(
        f"[OK] {len(manifest['records'])} team records"
    )
    print(
        f"[OK] Showdown commit: "
        f"{manifest.get('showdown_commit')}"
    )
    print(
        f"[OK] Showdown dirty: "
        f"{manifest.get('showdown_dirty')}"
    )
    print(
        "[OK] All packed-team SHA-256 hashes match"
    )


if __name__ == "__main__":
    main()
