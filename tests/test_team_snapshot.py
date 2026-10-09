import unittest

from pokebench30.evaluation.information import team_snapshot


class DummyMon:
    def __init__(self, hp=1.0, fainted=False):
        self.current_hp_fraction = hp
        self.fainted = fainted


class TeamSnapshotTests(unittest.TestCase):
    def test_snapshot_counts_only_objects_present_in_view(self):
        view = {
            "a": DummyMon(hp=0.5),
            "b": DummyMon(hp=0.0, fainted=True),
        }
        known, remaining, fainted, hp_sum = team_snapshot(view)
        self.assertEqual(known, 2)
        self.assertEqual(remaining, 1)
        self.assertEqual(fainted, 1)
        self.assertAlmostEqual(hp_sum, 0.5)


if __name__ == "__main__":
    unittest.main()
