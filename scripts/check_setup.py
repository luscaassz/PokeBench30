from __future__ import annotations

import socket
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ok(message: str) -> None:
    print(f"[OK] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def main() -> int:
    status = 0

    if sys.version_info >= (3, 10):
        ok(f"Python compatível: {sys.version.split()[0]}")
    else:
        fail(
            f"Python {sys.version.split()[0]} detectado; "
            "use Python 3.10 ou superior."
        )
        status = 1

    try:
        import poke_env  # noqa: F401
        ok("poke-env importado")
    except ImportError:
        fail("poke-env não instalado. Rode: pip install -r requirements.txt")
        status = 1

    try:
        import yaml  # noqa: F401
        ok("PyYAML importado")
    except ImportError:
        fail("PyYAML não instalado. Rode: pip install -r requirements.txt")
        status = 1

    config = PROJECT_ROOT / "config" / "generations.yaml"
    if config.exists():
        ok("generations.yaml encontrado")
    else:
        fail(f"Arquivo ausente: {config}")
        status = 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.5)
    try:
        sock.connect(("127.0.0.1", 8000))
        ok("Pokémon Showdown respondeu em localhost:8000")
    except OSError:
        fail(
            "Nenhum servidor respondeu em localhost:8000. "
            "Inicie o Pokémon Showdown com --no-security."
        )
        status = 1
    finally:
        sock.close()

    if status == 0:
        print("\nAmbiente pronto para o smoke test.")
    else:
        print("\nCorrija os itens [FAIL] antes de rodar o piloto.")

    return status


if __name__ == "__main__":
    raise SystemExit(main())
