import unittest

from pokebench30.teams.fixed_pool import load_fixed_config


class FixedTeamConfigTests(unittest.TestCase):
    def test_expected_generations_and_unique_pairs(self):
        config = load_fixed_config()
        generations = {
            int(key) for key in config["generations"]
        }
        self.assertEqual(generations, {1, 4, 9})

        for _, gen_cfg in config["generations"].items():
            pair_ids = [pair["id"] for pair in gen_cfg["pairs"]]
            self.assertEqual(len(pair_ids), len(set(pair_ids)))
            self.assertGreaterEqual(len(pair_ids), 3)

    def test_seeds_are_explicit(self):
        config = load_fixed_config()
        for _, gen_cfg in config["generations"].items():
            for pair in gen_cfg["pairs"]:
                self.assertIn("team_a_seed", pair)
                self.assertIn("team_b_seed", pair)
                self.assertNotEqual(
                    pair["team_a_seed"],
                    pair["team_b_seed"],
                )


if __name__ == "__main__":
    unittest.main()
