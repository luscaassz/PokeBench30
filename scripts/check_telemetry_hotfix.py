from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pokebench30.evaluation.telemetry import (
    canonical_fixed_pool_catalog_rows,
    fixed_pool_catalog_from_manifest,
    generation_dex_catalog,
    move_semantics,
    validate_fixed_pool_catalog_rows,
)
from pokebench30.rl.transfer_study import _write_fixed_pool_catalogs
from pokebench30.teams.fixed_pool import DEFAULT_POOL_ROOT, load_pool_manifest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate PokéBench30 v0.6.1 telemetry hotfix2."
    )
    parser.add_argument("--pool-root", default=str(DEFAULT_POOL_ROOT))
    args = parser.parse_args()

    try:
        from poke_env.battle.move import Move
    except ImportError as exc:
        raise SystemExit(
            "poke-env is required. Activate the same environment used by PokéBench30."
        ) from exc

    # 1) Hotfix1 semantic correction remains valid.
    swords_dance = move_semantics(Move("swordsdance", gen=4))
    if not swords_dance["feature_self_boost"]:
        raise SystemExit("[FAIL] Swords Dance is not classified as self_boost.")
    if swords_dance["feature_target_boost"]:
        raise SystemExit("[FAIL] Swords Dance is still classified as target_boost.")
    print("[OK] Swords Dance -> self_boost=True, target_boost=False")

    # 2) Hotfix1 generation-wide canonical base-species catalog remains valid.
    pokemon, _ = generation_dex_catalog([1, 4, 9])
    by_gen = {
        generation: [row for row in pokemon if int(row["generation"]) == generation]
        for generation in (1, 4, 9)
    }
    expected = {1: 151, 4: 493, 9: 1025}
    forbidden = {
        1: {"aerodactylmega", "arcaninehisui", "articunogalar", "arghonaut"},
        4: {"aerodactylmega", "arcaninehisui", "articunogalar", "arghonaut"},
    }
    for generation, rows in by_gen.items():
        ids = {str(row["species_id"]) for row in rows}
        count = len(ids)
        if count != expected[generation]:
            raise SystemExit(
                f"[FAIL] Gen {generation} base-species catalog: expected "
                f"{expected[generation]}, found {count}."
            )
        bad = sorted(ids & forbidden.get(generation, set()))
        if bad:
            raise SystemExit(
                f"[FAIL] Gen {generation} contains invalid future/CAP entries: {bad}"
            )
        print(f"[OK] Gen {generation} base-species catalog: {count} canonical entries")

    # 3) Immutable packed-team catalog itself must be exactly 10x2x6.
    pool_root = Path(args.pool_root).resolve()
    manifest = load_pool_manifest(pool_root)
    static_pokemon, static_moves = fixed_pool_catalog_from_manifest(
        manifest,
        project_root=PROJECT_ROOT,
    )
    static_pokemon, static_moves = canonical_fixed_pool_catalog_rows(
        static_pokemon,
        static_moves,
    )
    try:
        validate_fixed_pool_catalog_rows(static_pokemon, static_moves)
    except ValueError as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc

    for generation in (1, 4, 9):
        gen_rows = [
            row for row in static_pokemon
            if int(row["generation"]) == generation
        ]
        print(
            f"[OK] Gen {generation} fixed pool: "
            f"20 teams x 6 Pokémon = {len(gen_rows)} rows"
        )

    pokemon_keys = [
        (
            int(row["generation"]),
            str(row["pair_id"]),
            str(row["side"]),
            int(row["team_slot"]),
        )
        for row in static_pokemon
    ]
    if len(pokemon_keys) != len(set(pokemon_keys)):
        raise SystemExit("[FAIL] Duplicate Pokémon slot keys found in static pool.")

    move_keys = [
        (
            int(row["generation"]),
            str(row["pair_id"]),
            str(row["side"]),
            int(row["team_slot"]),
            int(row["move_slot"]),
        )
        for row in static_moves
    ]
    if len(move_keys) != len(set(move_keys)):
        raise SystemExit("[FAIL] Duplicate move-slot keys found in static pool.")
    print(f"[OK] Static fixed-pool move slots: {len(static_moves)} unique rows")

    # 4) Reproduce the exact hotfix1 aggregate failure: contaminate pair01-03
    # with post-battle slots 1-12. The final fixed-pool output must still use
    # only packed_fixed_team rows and remain 120 Pokémon/gen.
    dynamic_pokemon: list[dict] = []
    dynamic_moves: list[dict] = []
    for generation in (1, 4, 9):
        for pair_no in range(1, 4):
            pair_id = f"pair{pair_no:02d}"
            for side in ("A", "B"):
                for slot in range(1, 13):
                    dynamic_pokemon.append({
                        "generation": generation,
                        "evaluation_generation": generation,
                        "pair_id": pair_id,
                        "side": side,
                        "team_id": f"observed_{pair_id}_{side}",
                        "team_slot": slot,
                        "species": f"observed_species_{slot}",
                        "type_1": "PSYCHIC",
                        "type_2": "",
                        "catalog_source": "battle_observation",
                    })
                    for move_slot in range(1, 5):
                        dynamic_moves.append({
                            "generation": generation,
                            "evaluation_generation": generation,
                            "pair_id": pair_id,
                            "side": side,
                            "team_id": f"observed_{pair_id}_{side}",
                            "team_slot": slot,
                            "species": f"observed_species_{slot}",
                            "move_slot": move_slot,
                            "move_id": f"observed_move_{slot}_{move_slot}",
                            "move_type": "PSYCHIC",
                            "move_category": "STATUS",
                            "catalog_source": "battle_observation",
                        })

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        _write_fixed_pool_catalogs(
            out,
            static_pokemon + dynamic_pokemon,
            static_moves + dynamic_moves,
        )
        aggregate_pokemon = _read_csv(out / "fixed_pool_pokemon_catalog.csv")
        aggregate_moves = _read_csv(out / "fixed_pool_move_catalog.csv")

    for generation in (1, 4, 9):
        gen_rows = [
            row for row in aggregate_pokemon
            if int(row["generation"]) == generation
        ]
        if len(gen_rows) != 120:
            raise SystemExit(
                f"[FAIL] Gen {generation} contaminated aggregate still has "
                f"{len(gen_rows)} Pokémon rows; expected 120."
            )
        if max(int(row["team_slot"]) for row in gen_rows) > 6:
            raise SystemExit(
                f"[FAIL] Gen {generation} aggregate still contains team slots > 6."
            )
    if aggregate_pokemon and any(
        row.get("catalog_source") != "packed_fixed_team"
        for row in aggregate_pokemon
    ):
        raise SystemExit("[FAIL] Battle-observation rows leaked into fixed-pool catalog.")
    if aggregate_moves and any(
        row.get("catalog_source") != "packed_fixed_team"
        for row in aggregate_moves
    ):
        raise SystemExit("[FAIL] Battle-observation moves leaked into fixed-pool catalog.")

    print("[OK] Aggregate contamination regression: static packed teams win exclusively")
    print("[OK] No team_slot 7-12 leakage into fixed-pool structural catalogs")
    print("[OK] PokéBench30 v0.6.1 telemetry hotfix2 is ready.")


if __name__ == "__main__":
    main()
