import json
import tempfile
import unittest
from pathlib import Path

from pokebench30.evaluation.telemetry import (
    _canonical_manifest_records,
    _is_base_species_catalog_entry,
    behavior_summary,
    move_semantics,
    move_type_usage,
    parse_packed_team,
    write_telemetry_outputs,
)


class DummyEnum:
    def __init__(self, name):
        self.name = name


class DummyMove:
    def __init__(self, move_id, *, category, move_type, base_power=0, boosts=None,
                 self_boost=None, status=None, side_condition=None, priority=0,
                 heal=None, drain=None, recoil=None, self_switch=None, target="NORMAL"):
        self.id = move_id
        self.name = move_id
        self.category = DummyEnum(category)
        self.type = DummyEnum(move_type)
        self.base_power = base_power
        self.accuracy = 1.0
        self.boosts = boosts or {}
        self.self_boost = self_boost or {}
        self.status = DummyEnum(status) if status else None
        self.volatile_status = None
        self.side_condition = DummyEnum(side_condition) if side_condition else None
        self.priority = priority
        self.heal = heal
        self.drain = drain
        self.recoil = recoil
        self.self_switch = self_switch
        self.force_switch = False
        self.target = DummyEnum(target)
        self.weather = None
        self.terrain = None


class TelemetryTests(unittest.TestCase):
    def test_move_semantics_is_multilabel(self):
        move = DummyMove(
            "drainpunch",
            category="PHYSICAL",
            move_type="FIGHTING",
            base_power=75,
            boosts={"def": -1},
            status="PAR",
            drain=0.5,
            priority=1,
        )
        row = move_semantics(move)
        self.assertTrue(row["feature_damaging"])
        self.assertTrue(row["feature_target_debuff"])
        self.assertTrue(row["feature_status_inflict"])
        self.assertTrue(row["feature_drain"])
        self.assertTrue(row["feature_priority"])


    def test_self_target_boost_is_not_target_boost(self):
        swords_dance = DummyMove(
            "swordsdance",
            category="STATUS",
            move_type="NORMAL",
            boosts={"atk": 2},
            target="SELF",
        )
        row = move_semantics(swords_dance)
        self.assertTrue(row["feature_self_boost"])
        self.assertFalse(row["feature_target_boost"])
        self.assertTrue(row["feature_any_buff"])
        self.assertEqual(json.loads(row["self_boosts_json"]), {"atk": 2})
        self.assertEqual(json.loads(row["boosts_json"]), {})

    def test_target_debuff_remains_target_debuff(self):
        growl = DummyMove(
            "growl",
            category="STATUS",
            move_type="NORMAL",
            boosts={"atk": -1},
            target="NORMAL",
        )
        row = move_semantics(growl)
        self.assertFalse(row["feature_self_debuff"])
        self.assertTrue(row["feature_target_debuff"])
        self.assertTrue(row["feature_any_debuff"])

    def test_generation_base_species_filter_removes_future_forms_and_cap(self):
        self.assertTrue(_is_base_species_catalog_entry(
            "bulbasaur",
            {"num": 1, "name": "Bulbasaur", "baseSpecies": "bulbasaur"},
            1,
        ))
        self.assertFalse(_is_base_species_catalog_entry(
            "aerodactylmega",
            {
                "num": 142,
                "name": "Aerodactyl-Mega",
                "baseSpecies": "Aerodactyl",
                "forme": "Mega",
            },
            1,
        ))
        self.assertFalse(_is_base_species_catalog_entry(
            "arcaninehisui",
            {
                "num": 59,
                "name": "Arcanine-Hisui",
                "baseSpecies": "Arcanine",
                "forme": "Hisui",
            },
            4,
        ))
        self.assertFalse(_is_base_species_catalog_entry(
            "arghonaut",
            {"num": -2, "name": "Arghonaut", "baseSpecies": "arghonaut"},
            4,
        ))
        self.assertFalse(_is_base_species_catalog_entry(
            "chikorita",
            {"num": 152, "name": "Chikorita", "baseSpecies": "chikorita"},
            1,
        ))

    def test_generation_catalog_removes_cosmetic_aliases(self):
        raw = {"num": 201, "name": "Unown", "baseSpecies": "unown"}
        self.assertTrue(_is_base_species_catalog_entry("unown", raw, 4))
        self.assertFalse(_is_base_species_catalog_entry("unownb", raw, 4))

    def test_manifest_catalog_uses_same_last_record_wins_rule_as_evaluator(self):
        manifest = {
            "records": [
                {"generation": 1, "pair_id": "pair01", "side": "A", "team_id": "old"},
                {"generation": 1, "pair_id": "pair01", "side": "B", "team_id": "b"},
                {"generation": 1, "pair_id": "pair01", "side": "A", "team_id": "new"},
            ]
        }
        rows = _canonical_manifest_records(manifest)
        self.assertEqual(len(rows), 2)
        by_side = {row["side"]: row for row in rows}
        self.assertEqual(by_side["A"]["team_id"], "new")
        self.assertEqual(by_side["B"]["team_id"], "b")

    def test_availability_adjusted_selection(self):
        profiles = [
            {
                "move_id": "thunderbolt",
                "move_type": "ELECTRIC",
                "move_category": "SPECIAL",
                "feature_damaging": True,
                "feature_status": False,
            },
            {
                "move_id": "recover",
                "move_type": "NORMAL",
                "move_category": "STATUS",
                "feature_damaging": False,
                "feature_status": True,
            },
        ]
        # Fill unused feature flags so summary can read them safely.
        from pokebench30.evaluation.telemetry import MOVE_FEATURES
        for p in profiles:
            for feature in MOVE_FEATURES:
                p.setdefault(f"feature_{feature}", False)

        actions = [{
            "agent_role": "rl",
            "action_type": "move",
            "action": "recover",
            "move_type": "NORMAL",
            "move_category": "STATUS",
            "move_feature_status": True,
            "move_feature_damaging": False,
            "move_stab": 1.0,
            "move_type_effectiveness": 1.0,
            "move_base_power": 0,
            "move_accuracy": 1.0,
            "available_count_status": 1,
            "available_count_damaging": 1,
            "available_move_profiles_json": json.dumps(profiles),
        }]
        for feature in MOVE_FEATURES:
            actions[0].setdefault(f"move_feature_{feature}", False)
            actions[0].setdefault(
                f"available_count_{feature}",
                sum(bool(p[f"feature_{feature}"]) for p in profiles),
            )

        summary = behavior_summary(actions)
        self.assertEqual(summary["availability_rate_status"], 1.0)
        self.assertEqual(summary["selection_given_available_status"], 1.0)
        self.assertEqual(summary["selection_given_available_damaging"], 0.0)

        type_rows = {row["move_type"]: row for row in move_type_usage(actions)}
        self.assertEqual(type_rows["NORMAL"]["selection_given_available"], 1.0)
        self.assertEqual(type_rows["ELECTRIC"]["selection_given_available"], 0.0)

    def test_outputs_include_compressed_opportunities(self):
        actions = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_telemetry_outputs(
                root,
                actions=actions,
                battles=[],
                team_pokemon_rows=[],
                team_move_rows=[],
            )
            self.assertEqual(manifest["rl_decision_rows"], 0)
            self.assertTrue((root / "telemetry_move_opportunities.csv.gz").exists())

    def test_parse_packed_team_core_fields(self):
        packed = (
            "Pika|Pikachu|lightball|static|thunderbolt,volttackle|Jolly|||||50"
            "]Sto|Starmie|leftovers|naturalcure|surf,recover|Timid"
        )
        rows = parse_packed_team(packed)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["species_raw"], "Pikachu")
        self.assertEqual(rows[0]["move_ids"], ["thunderbolt", "volttackle"])
        self.assertEqual(rows[1]["species_raw"], "Starmie")


if __name__ == "__main__":
    unittest.main()
