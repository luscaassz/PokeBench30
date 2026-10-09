import unittest

from pokebench30.evaluation.mirrored import (
    build_controlled_summary,
    build_mirror_cycles,
)


class MirroredMetricsTests(unittest.TestCase):
    def _row(
        self,
        leg,
        repeat,
        result,
        agent_1_team,
        agent_2_team,
    ):
        return {
            "generation": 9,
            "pair_id": "pair01",
            "agent_1": "A",
            "agent_2": "B",
            "repeat_index": repeat,
            "mirror_leg": leg,
            "result_agent_1": result,
            "agent_1_team_id": agent_1_team,
            "agent_2_team_id": agent_2_team,
            "team_a_id": "teamA",
            "team_b_id": "teamB",
        }

    def test_split_mirror_is_neutral_cycle(self):
        rows = [
            self._row("AB", 1, "win", "teamA", "teamB"),
            self._row("BA", 1, "loss", "teamB", "teamA"),
        ]
        cycles = build_mirror_cycles(rows)
        self.assertEqual(cycles[0]["agent_1_points"], 1.0)
        self.assertEqual(cycles[0]["cycle_outcome"], "balanced")

    def test_two_wins_is_full_mirror_score(self):
        rows = [
            self._row("AB", 1, "win", "teamA", "teamB"),
            self._row("BA", 1, "win", "teamB", "teamA"),
        ]
        cycles = build_mirror_cycles(rows)
        summary = build_controlled_summary(rows, cycles)
        self.assertEqual(
            summary[0]["normalized_mirror_score_agent_1"],
            1.0,
        )

    def test_missing_leg_raises(self):
        rows = [
            self._row("AB", 1, "win", "teamA", "teamB"),
        ]
        with self.assertRaises(ValueError):
            build_mirror_cycles(rows)


if __name__ == "__main__":
    unittest.main()
