from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate a PokéBench30 v0.4 universal_states.jsonl dataset."
        )
    )
    parser.add_argument(
        "run_dir",
        help="Path to results/controlled_YYYYMMDD_HHMMSS",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    schema_path = run_dir / "observation_schema.json"
    states_path = run_dir / "universal_states.jsonl"

    if not schema_path.exists():
        raise SystemExit(
            f"[FAIL] Missing {schema_path}"
        )

    if not states_path.exists():
        raise SystemExit(
            f"[FAIL] Missing {states_path}"
        )

    schema = json.loads(
        schema_path.read_text(encoding="utf-8")
    )
    expected_dim = int(schema["observation_dim"])
    expected_action_dim = int(schema["action_dim"])
    expected_hash = schema["sha256"]

    records = 0
    mapped_actions = 0
    illegal_chosen_actions = 0
    schema_mismatches = 0
    dimension_errors = 0
    mask_errors = 0
    non_finite = 0

    generations = set()

    with states_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            records += 1

            generations.add(
                int(row.get("generation", 0))
            )

            if row.get("schema_sha256") != expected_hash:
                schema_mismatches += 1

            vector = row.get("vector", [])
            mask = row.get("legal_action_mask", [])

            if len(vector) != expected_dim:
                dimension_errors += 1

            if len(mask) != expected_action_dim:
                mask_errors += 1

            if any(
                not math.isfinite(float(value))
                for value in vector
            ):
                non_finite += 1

            chosen = int(
                row.get("chosen_action_index", -1)
            )

            if chosen >= 0:
                mapped_actions += 1

                if (
                    chosen >= len(mask)
                    or int(mask[chosen]) != 1
                ):
                    illegal_chosen_actions += 1

    print(f"[INFO] Records: {records}")
    print(
        f"[INFO] Generations: "
        f"{sorted(generations)}"
    )
    print(
        f"[INFO] Chosen actions mapped into UBR: "
        f"{mapped_actions}/{records}"
    )

    checks = {
        "schema mismatches": schema_mismatches,
        "dimension errors": dimension_errors,
        "mask errors": mask_errors,
        "non-finite vectors": non_finite,
        "chosen actions marked illegal": illegal_chosen_actions,
    }

    failed = False
    for name, count in checks.items():
        if count:
            print(f"[FAIL] {name}: {count}")
            failed = True
        else:
            print(f"[OK] {name}: 0")

    if failed:
        raise SystemExit(1)

    print(
        "\n[OK] Universal-state dataset passed structural validation."
    )


if __name__ == "__main__":
    main()
