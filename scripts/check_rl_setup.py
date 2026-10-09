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

        ok(f"PokéBench30 version: {__version__}")

        expected = {
            "ubr-1.1": (472, "276327ddc0883116305f6d564cb34ba43d7cf2eff8294185aeb9587c4485a20e"),
            "ubr-1.2": (518, "d4b3858aab3a1c3bb1e17d9f34becf93814be05bfc949d52023651035f20e4c3"),
        }
        for version, (dim, sha) in expected.items():
            schema = build_feature_schema(version)
            ok(
                f"{version}: observation={schema.observation_dim}, "
                f"actions={schema.action_dim}, sha256={schema.sha256}"
            )
            if schema.observation_dim != dim:
                fail(f"{version} expected observation dimension {dim}.")
                status = 1
            if schema.action_dim != 10:
                fail(f"{version} expected 10 actions.")
                status = 1
            if schema.sha256 != sha:
                fail(f"{version} schema hash drifted from the validated baseline.")
                status = 1

        old = build_feature_schema("ubr-1.1")
        new = build_feature_schema("ubr-1.2")
        if new.feature_names[: old.observation_dim] != old.feature_names:
            fail("UBR 1.2 no longer preserves the exact UBR 1.1 prefix.")
            status = 1
        else:
            ok("UBR 1.2 preserves the exact 472-feature UBR 1.1 prefix")

    except Exception as exc:
        fail(f"Could not validate PokéBench30 schemas: {exc}")
        status = 1

    if status:
        print("\nFix the [FAIL] items before RL training.")
        raise SystemExit(status)

    print("\n[OK] PokéBench30 v0.5.2 RL + multi-seed setup is ready.")


if __name__ == "__main__":
    main()
