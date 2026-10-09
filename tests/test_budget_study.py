import csv
import tempfile
import unittest
from pathlib import Path

from pokebench30.rl.budget_study import (
    BudgetEvaluationResult,
    _aggregate,
)


class BudgetStudyTests(unittest.TestCase):
    def make_result(
        self,
        *,
        budget,
        representation,
        seed,
        opponent,
        score,
    ):
        battles = 60
        wins = round(score * battles)
        return BudgetEvaluationResult(
            budget=budget,
            representation=representation,
            seed=seed,
            opponent=opponent,
            source="test",
            model_dir="model",
            evaluation_dir="eval",
            battles=battles,
            mirror_cycles=30,
            wins_rl=wins,
            wins_opponent=battles - wins,
            draws=0,
            raw_win_rate_rl=score,
            mirror_score_rl=score,
            team_assignment_gap=0.0,
        )

    def test_budget_gain_and_interaction_outputs(self):
        results = []
        seeds = [42, 123]
        values = {
            (10000, "ubr-1.1"): [0.60, 0.70],
            (10000, "ubr-1.2"): [0.60, 0.70],
            (50000, "ubr-1.1"): [0.70, 0.80],
            (50000, "ubr-1.2"): [0.80, 0.90],
        }

        for budget in (10000, 50000):
            for representation in ("ubr-1.1", "ubr-1.2"):
                for seed, score in zip(
                    seeds,
                    values[(budget, representation)],
                ):
                    results.append(
                        self.make_result(
                            budget=budget,
                            representation=representation,
                            seed=seed,
                            opponent="random",
                            score=score,
                        )
                    )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _aggregate(results, out)

            self.assertTrue((out / "aggregate_summary.csv").exists())
            self.assertTrue((out / "budget_comparison.csv").exists())
            self.assertTrue(
                (out / "budget_representation_interaction.csv").exists()
            )

            with (out / "budget_comparison.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))

            row_11 = next(
                row for row in rows
                if row["representation"] == "ubr-1.1"
            )
            row_12 = next(
                row for row in rows
                if row["representation"] == "ubr-1.2"
            )

            self.assertAlmostEqual(float(row_11["mean_gain"]), 0.10)
            self.assertAlmostEqual(float(row_12["mean_gain"]), 0.20)

            with (out / "budget_representation_interaction.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                interaction = list(csv.DictReader(handle))

            self.assertEqual(len(interaction), 1)
            self.assertAlmostEqual(
                float(
                    interaction[0][
                        "difference_of_gains_ubr_1_2_minus_1_1"
                    ]
                ),
                0.10,
            )


if __name__ == "__main__":
    unittest.main()
