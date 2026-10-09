import csv
import tempfile
import unittest
from pathlib import Path

from pokebench30.rl.curriculum_study import (
    EvalRow,
    Stage,
    _write_aggregates,
    checkpoints,
    opponent_at,
    validate_regimes,
)


class CurriculumScheduleTests(unittest.TestCase):
    def setUp(self):
        self.regimes = {
            "direct_type_aware": [Stage("type_aware", 300000)],
            "random_to_type_aware": [
                Stage("random", 100000),
                Stage("type_aware", 200000),
            ],
            "progressive": [
                Stage("random", 100000),
                Stage("max_damage", 100000),
                Stage("type_aware", 100000),
            ],
            "random_only": [Stage("random", 300000)],
        }

    def test_default_schedule_is_valid(self):
        validate_regimes(
            self.regimes,
            total_timesteps=300000,
            checkpoint_interval=50000,
        )
        self.assertEqual(
            checkpoints(300000, 50000),
            [50000, 100000, 150000, 200000, 250000, 300000],
        )

    def test_progressive_opponents(self):
        regime = self.regimes["progressive"]
        self.assertEqual(opponent_at(regime, 0), "random")
        self.assertEqual(opponent_at(regime, 99999), "random")
        self.assertEqual(opponent_at(regime, 100000), "max_damage")
        self.assertEqual(opponent_at(regime, 199999), "max_damage")
        self.assertEqual(opponent_at(regime, 200000), "type_aware")

    def test_misaligned_transition_is_rejected(self):
        bad = {
            "bad": [Stage("random", 75000), Stage("type_aware", 225000)]
        }
        with self.assertRaises(ValueError):
            validate_regimes(
                bad,
                total_timesteps=300000,
                checkpoint_interval=50000,
            )


class CurriculumAggregationTests(unittest.TestCase):
    def _row(self, regime, seed, opponent, score):
        battles = 60
        wins = round(battles * score)
        return EvalRow(
            regime=regime,
            seed=seed,
            checkpoint=300000,
            opponent=opponent,
            evaluation_kind="final",
            repeats_per_leg=10,
            battles=battles,
            mirror_cycles=30,
            wins_rl=wins,
            wins_opponent=battles - wins,
            draws=0,
            raw_win_rate_rl=score,
            mirror_score_rl=score,
            team_assignment_gap=0.0,
            model_dir="model",
            evaluation_dir="eval",
        )

    def test_curriculum_comparison_against_direct(self):
        rows = [
            self._row("direct_type_aware", 42, "type_aware", 0.10),
            self._row("direct_type_aware", 123, "type_aware", 0.20),
            self._row("progressive", 42, "type_aware", 0.30),
            self._row("progressive", 123, "type_aware", 0.40),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_aggregates(root, rows, [])
            path = root / "aggregate" / "curriculum_comparison.csv"
            with path.open("r", encoding="utf-8", newline="") as handle:
                data = list(csv.DictReader(handle))
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["regime"], "progressive")
            self.assertAlmostEqual(
                float(data[0]["mean_difference_vs_direct"]), 0.20
            )


if __name__ == "__main__":
    unittest.main()
