import csv
import tempfile
import unittest
from pathlib import Path

from pokebench30.evaluation.telemetry import (
    canonical_fixed_pool_catalog_rows,
    validate_fixed_pool_catalog_rows,
)
from pokebench30.rl.transfer_study import _write_fixed_pool_catalogs


class TelemetryHotfixTests(unittest.TestCase):
    def _static_rows(self):
        pokemon_rows = []
        move_rows = []
        for generation in (1, 4, 9):
            for pair_no in range(1, 11):
                pair_id = f"pair{pair_no:02d}"
                for side in ("A", "B"):
                    team_id = f"gen{generation}_{pair_id}_{side}"
                    for slot in range(1, 7):
                        pokemon_rows.append({
                            "generation": generation,
                            "pair_id": pair_id,
                            "side": side,
                            "team_id": team_id,
                            "team_slot": slot,
                            "species": f"species{generation}_{pair_id}_{side}_{slot}",
                            "type_1": "NORMAL",
                            "type_2": "",
                            "catalog_source": "packed_fixed_team",
                        })
                        for move_slot in range(1, 5):
                            move_rows.append({
                                "generation": generation,
                                "pair_id": pair_id,
                                "side": side,
                                "team_id": team_id,
                                "team_slot": slot,
                                "species": f"species{generation}_{pair_id}_{side}_{slot}",
                                "move_slot": move_slot,
                                "move_id": f"move{generation}_{pair_id}_{side}_{slot}_{move_slot}",
                                "move_type": "NORMAL",
                                "move_category": "PHYSICAL",
                                "feature_damaging": True,
                                "catalog_source": "packed_fixed_team",
                            })
        return pokemon_rows, move_rows

    def _contaminating_battle_rows(self):
        pokemon_rows = []
        move_rows = []
        # Reproduce the observed smoke failure: pair01-03 acquired extra
        # post-battle slots 7-12 and transformed state.
        for generation in (1, 4, 9):
            for pair_no in range(1, 4):
                pair_id = f"pair{pair_no:02d}"
                for side in ("A", "B"):
                    for slot in range(1, 13):
                        pokemon_rows.append({
                            "generation": generation,
                            "evaluation_generation": generation,
                            "pair_id": pair_id,
                            "side": side,
                            "team_id": f"observed_{pair_id}_{side}",
                            "team_slot": slot,
                            "species": f"observed_species_{slot}",
                            "type_1": "PSYCHIC" if slot == 1 else "NORMAL",
                            "type_2": "",
                            "catalog_source": "battle_observation",
                        })
                        for move_slot in range(1, 5):
                            move_rows.append({
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
                                "feature_damaging": False,
                                "catalog_source": "battle_observation",
                            })
        return pokemon_rows, move_rows

    def test_static_source_wins_over_battle_observation_rows(self):
        static_pokemon, static_moves = self._static_rows()
        dynamic_pokemon, dynamic_moves = self._contaminating_battle_rows()

        pokemon, moves = canonical_fixed_pool_catalog_rows(
            static_pokemon + dynamic_pokemon,
            static_moves + dynamic_moves,
        )

        self.assertEqual(len(pokemon), 360)
        self.assertEqual(len(moves), 1440)
        self.assertTrue(
            all(row["catalog_source"] == "packed_fixed_team" for row in pokemon)
        )
        self.assertTrue(
            all(row["catalog_source"] == "packed_fixed_team" for row in moves)
        )

        for generation in (1, 4, 9):
            self.assertEqual(
                sum(int(row["generation"]) == generation for row in pokemon),
                120,
            )

        validate_fixed_pool_catalog_rows(pokemon, moves)

    def test_writer_cannot_reintroduce_slots_7_to_12(self):
        static_pokemon, static_moves = self._static_rows()
        dynamic_pokemon, dynamic_moves = self._contaminating_battle_rows()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixed_pool_catalogs(
                root,
                static_pokemon + dynamic_pokemon,
                static_moves + dynamic_moves,
            )

            with (root / "fixed_pool_pokemon_catalog.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                pokemon = list(csv.DictReader(handle))
            with (root / "fixed_pool_move_catalog.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                moves = list(csv.DictReader(handle))

            self.assertEqual(len(pokemon), 360)
            self.assertEqual(len(moves), 1440)
            self.assertLessEqual(max(int(row["team_slot"]) for row in pokemon), 6)
            self.assertLessEqual(max(int(row["move_slot"]) for row in moves), 4)

            for generation in (1, 4, 9):
                gen_pokemon = [
                    row for row in pokemon
                    if int(row["generation"]) == generation
                ]
                self.assertEqual(len(gen_pokemon), 120)

    def test_duplicate_keys_fail_fast(self):
        static_pokemon, static_moves = self._static_rows()
        duplicate = dict(static_pokemon[0])
        pokemon = static_pokemon + [duplicate]
        with self.assertRaises(ValueError):
            validate_fixed_pool_catalog_rows(pokemon, static_moves)


if __name__ == "__main__":
    unittest.main()
