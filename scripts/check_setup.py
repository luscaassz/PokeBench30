from __future__ import annotations

import importlib.metadata
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ok(message: str) -> None: print(f"[OK] {message}")
def fail(message: str) -> None: print(f"[FAIL] {message}")


def main() -> int:
    status = 0
    if sys.version_info >= (3, 10): ok(f"Python compatible: {sys.version.split()[0]}")
    else:
        fail(f"Python {sys.version.split()[0]} detected; use Python 3.10+.")
        status = 1

    try:
        import poke_env  # noqa: F401
        try: version = importlib.metadata.version("poke-env")
        except importlib.metadata.PackageNotFoundError: version = "unknown"
        ok(f"poke-env imported (version {version})")
    except ImportError:
        fail("poke-env not installed. Run: python -m pip install -r requirements.txt")
        status = 1

    try:
        import yaml  # noqa: F401
        ok("PyYAML imported")
    except ImportError:
        fail("PyYAML not installed. Run: python -m pip install -r requirements.txt")
        status = 1

    try:
        import matplotlib  # noqa: F401
        ok("matplotlib imported")
    except ImportError:
        fail("matplotlib not installed. Run: python -m pip install -r requirements.txt")
        status = 1

    try:
        import numpy  # noqa: F401
        ok("NumPy imported")
    except ImportError:
        fail("NumPy not installed. Run: python -m pip install -r requirements.txt")
        status = 1

    config = PROJECT_ROOT / "config" / "generations.yaml"
    if config.exists(): ok("generations.yaml found")
    else:
        fail(f"Missing file: {config}")
        status = 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.5)
    try:
        sock.connect(("127.0.0.1", 8000))
        ok("Pokemon Showdown responded at localhost:8000")
    except OSError:
        fail("No server responded at localhost:8000. Start Pokemon Showdown with --no-security.")
        status = 1
    finally:
        sock.close()

    print("\nEnvironment ready for PokéBench30 v0.4.1." if status == 0 else "\nFix the [FAIL] items before running v0.4.1 experiments.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
