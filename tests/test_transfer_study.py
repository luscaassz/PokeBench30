import csv
import tempfile
import unittest
from pathlib import Path

from pokebench30.rl.transfer_study import TransferScore, build_score_outputs


class TransferStudyTests(unittest.TestCase):
    def make_score(self, train, evaluate, score):
        return TransferScore(
            training_generation=train,
            evaluation_generation=evaluate,
            seed=42,
            opponent="type_aware",
            battles=60,
            mirror_cycles=30,
            wins_rl=int(round(score * 60)),
            wins_opponent=60 - int(round(score * 60)),
            draws=0,
            mirror_score_rl=score,
            raw_win_rate_rl=score,
            team_assignment_gap=0.0,
            evaluation_dir="x",
            model_dir="y",
        )

    def test_transfer_score_schema(self):
        row = self.make_score(1, 9, 1 / 6)
        self.assertEqual(row.training_generation, 1)
        self.assertEqual(row.evaluation_generation, 9)
        self.assertEqual(row.battles, 60)

    def test_transfer_gap_uses_same_seed_in_domain_baseline(self):
        scores = []
        values = {
            (1, 1): 0.8, (1, 4): 0.3, (1, 9): 0.2,
            (4, 1): 0.5, (4, 4): 0.7, (4, 9): 0.4,
            (9, 1): 0.4, (9, 4): 0.5, (9, 9): 0.6,
        }
        for (train, evaluate), score in values.items():
            scores.append(self.make_score(train, evaluate, score))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_score_outputs(root, scores)
            path = root / "aggregate" / "transfer_gap_by_seed.csv"
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        target = next(
            row for row in rows
            if row["training_generation"] == "1"
            and row["evaluation_generation"] == "9"
        )
        self.assertAlmostEqual(float(target["transfer_score"]), 0.2)
        self.assertAlmostEqual(float(target["in_domain_score"]), 0.6)
        self.assertAlmostEqual(float(target["transfer_gap"]), -0.4)


if __name__ == "__main__":
    unittest.main()
