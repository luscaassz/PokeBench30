import unittest

from pokebench30.evaluation.information import final_information_snapshot


class DummyMon:
    def __init__(self, hp=1.0, fainted=False):
        self.current_hp_fraction = hp
        self.fainted = fainted


class DummyBattle:
    def __init__(self, team, opponent_team):
        self.team = team
        self.opponent_team = opponent_team


class InformationPolicyTests(unittest.TestCase):
    def test_hidden_opponent_slots_stay_hidden_in_observation(self):
        # Player 1 owns six Pokémon but has only revealed two of player 2's.
        p1_team = {f"p1_{i}": DummyMon() for i in range(6)}
        p2_team = {f"p2_{i}": DummyMon() for i in range(6)}

        p1_seen_opponent = {
            "p2_0": DummyMon(hp=0.4),
            "p2_1": DummyMon(hp=0.0, fainted=True),
        }
        p2_seen_opponent = {
            "p1_0": DummyMon(hp=0.6),
            "p1_1": DummyMon(hp=0.8),
            "p1_2": DummyMon(hp=1.0),
        }

        p1 = DummyBattle(p1_team, p1_seen_opponent)
        p2 = DummyBattle(p2_team, p2_seen_opponent)

        info = final_information_snapshot(p1, p2)

        # Ground truth exists for evaluator-only metrics.
        self.assertEqual(info["agent_1_team_size_true"], 6)
        self.assertEqual(info["agent_2_team_size_true"], 6)

        # But each observation keeps only what was actually revealed.
        self.assertEqual(info["agent_1_opponent_revealed"], 2)
        self.assertEqual(info["agent_2_opponent_revealed"], 3)
        self.assertAlmostEqual(info["agent_1_opponent_reveal_fraction"], 2 / 6, places=6)
        self.assertAlmostEqual(info["agent_2_opponent_reveal_fraction"], 3 / 6, places=6)


if __name__ == "__main__":
    unittest.main()
