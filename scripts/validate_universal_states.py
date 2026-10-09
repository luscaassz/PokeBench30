from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
import math


def _read_action_lookup(run_dir: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    actions_path = run_dir / "actions.csv"
    if not actions_path.exists():
        return {}

    with actions_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    lookup: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row.get("battle_tag", ""),
            int(float(row.get("request_index", 0) or 0)),
            row.get("agent", ""),
        )
        lookup[key] = row
    return lookup


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate a PokéBench30 v0.5.1 UBR 1.2 universal_states.jsonl dataset."
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
        raise SystemExit(f"[FAIL] Missing {schema_path}")
    if not states_path.exists():
        raise SystemExit(f"[FAIL] Missing {states_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    expected_dim = int(schema["observation_dim"])
    expected_action_dim = int(schema["action_dim"])
    expected_hash = schema["sha256"]

    records = 0
    mapped_actions = 0
    illegal_chosen_actions = 0
    unmapped_actions = 0
    schema_mismatches = 0
    dimension_errors = 0
    mask_errors = 0
    empty_legal_masks = 0
    non_finite = 0
    context_value_errors = 0
    context_nonzero_records = 0
    context_activation_counts: dict[str, int] = {}

    generations = set()
    by_generation: dict[int, dict[str, int]] = {}
    unmapped_examples: list[dict] = []
    action_lookup = _read_action_lookup(run_dir)

    with states_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            records += 1
            generation = int(row.get("generation", 0))
            generations.add(generation)
            by_generation.setdefault(generation, {"records": 0, "mapped": 0})
            by_generation[generation]["records"] += 1

            if row.get("schema_sha256") != expected_hash:
                schema_mismatches += 1

            vector = row.get("vector", [])
            mask = row.get("legal_action_mask", [])

            if len(vector) != expected_dim:
                dimension_errors += 1
            if len(mask) != expected_action_dim:
                mask_errors += 1

            if any(not math.isfinite(float(value)) for value in vector):
                non_finite += 1

            # UBR 1.2 battle-context features are binary except normalized
            # Spikes / Toxic Spikes layer counts.
            feature_names = schema.get("feature_names", [])
            if len(vector) == len(feature_names):
                row_has_context = False
                for index, name in enumerate(feature_names):
                    if not name.startswith("battle_context."):
                        continue
                    value = float(vector[index])
                    if value > 1e-8:
                        row_has_context = True
                        context_activation_counts[name] = (
                            context_activation_counts.get(name, 0) + 1
                        )
                    if name.endswith(".spikes") or name.endswith(".toxic_spikes"):
                        if value < -1e-8 or value > 1.0 + 1e-8:
                            context_value_errors += 1
                    elif not (
                        abs(value) <= 1e-8
                        or abs(value - 1.0) <= 1e-8
                    ):
                        context_value_errors += 1
                if row_has_context:
                    context_nonzero_records += 1

            if len(mask) == expected_action_dim and sum(int(v) for v in mask) == 0:
                empty_legal_masks += 1

            chosen = int(row.get("chosen_action_index", -1))
            if chosen >= 0:
                mapped_actions += 1
                by_generation[generation]["mapped"] += 1
                if chosen >= len(mask) or int(mask[chosen]) != 1:
                    illegal_chosen_actions += 1
            else:
                unmapped_actions += 1
                if len(unmapped_examples) < 12:
                    key = (
                        row.get("battle_tag", ""),
                        int(row.get("request_index", 0) or 0),
                        row.get("agent", ""),
                    )
                    action_row = action_lookup.get(key, {})
                    unmapped_examples.append({
                        "generation": generation,
                        "agent": row.get("agent", ""),
                        "battle_tag": row.get("battle_tag", ""),
                        "request_index": row.get("request_index", ""),
                        "turn": row.get("turn", ""),
                        "action": action_row.get("action", "unknown"),
                        "action_type": action_row.get("action_type", "unknown"),
                    })

    print(f"[INFO] Records: {records}")
    print(f"[INFO] Generations: {sorted(generations)}")
    print(f"[INFO] Chosen actions mapped into UBR: {mapped_actions}/{records}")

    for generation in sorted(by_generation):
        stats = by_generation[generation]
        print(
            f"[INFO] Gen {generation}: "
            f"{stats['mapped']}/{stats['records']} actions mapped"
        )

    print(
        f"[INFO] Records with non-zero UBR 1.2 battle context: "
        f"{context_nonzero_records}/{records}"
    )
    if context_activation_counts:
        top_context = sorted(
            context_activation_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:12]
        print("[INFO] Most-observed battle-context features:")
        for name, count in top_context:
            print(f"  {name}: {count}")

    checks = {
        "schema mismatches": schema_mismatches,
        "dimension errors": dimension_errors,
        "mask errors": mask_errors,
        "non-finite vectors": non_finite,
        "battle-context value errors": context_value_errors,
        "empty legal-action masks": empty_legal_masks,
        "unmapped chosen actions": unmapped_actions,
        "chosen actions marked illegal": illegal_chosen_actions,
    }

    failed = False
    for name, count in checks.items():
        if count:
            print(f"[FAIL] {name}: {count}")
            failed = True
        else:
            print(f"[OK] {name}: 0")

    if unmapped_examples:
        print("\n[INFO] Unmapped-action examples:")
        for item in unmapped_examples:
            print(
                "  "
                f"Gen {item['generation']} | {item['agent']} | "
                f"turn={item['turn']} | request={item['request_index']} | "
                f"{item['action_type']}={item['action']} | "
                f"{item['battle_tag']}"
            )

    if failed:
        raise SystemExit(1)

    print("\n[OK] Universal-state dataset passed strict UBR validation.")


if __name__ == "__main__":
    main()
