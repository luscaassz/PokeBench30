from __future__ import annotations

from pathlib import Path
import importlib
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def ok(message: str) -> None:
    print(f"[OK] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def main() -> None:
    status = 0

    packages = [
        ("poke_env", "poke-env"),
        ("gymnasium", "gymnasium"),
        ("stable_baselines3", "stable-baselines3"),
        ("sb3_contrib", "sb3-contrib"),
        ("torch", "torch"),
        ("tensorboard", "tensorboard"),
        ("numpy", "numpy"),
        ("yaml", "PyYAML"),
    ]

    for module_name, package_name in packages:
        try:
            importlib.import_module(module_name)
            ok(f"{package_name} imported")
        except ImportError:
            fail(
                f"{package_name} not installed. Run: "
                "python -m pip install -r requirements.txt"
            )
            status = 1

    try:
        from pokebench30 import __version__
        from pokebench30.representation.schema import build_feature_schema

        schema = build_feature_schema()
        ok(f"PokéBench30 version: {__version__}")
        ok(f"UBR schema: {schema.version}")
        ok(f"Observation dimension: {schema.observation_dim}")
        ok(f"Action dimension: {schema.action_dim}")

        if schema.version != "ubr-1.2":
            fail(
                f"v0.5.1 expects UBR 1.2, found {schema.version}"
            )
            status = 1

        if schema.observation_dim != 518:
            fail(
                f"v0.5.1 expects 518 observations, found "
                f"{schema.observation_dim}"
            )
            status = 1

        if schema.action_dim != 10:
            fail(
                f"v0.5.1 expects 10 actions, found {schema.action_dim}"
            )
            status = 1

    except Exception as exc:
        fail(f"Could not validate PokéBench30 schema: {exc}")
        status = 1

    if status:
        print("\nFix the [FAIL] items before RL training.")
        raise SystemExit(status)

    print("\n[OK] PokéBench30 v0.5.1 RL setup is ready.")


if __name__ == "__main__":
    main()
