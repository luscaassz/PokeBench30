import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class RLConfigTests(unittest.TestCase):
    def test_rl_baseline_contract(self):
        config = yaml.safe_load(
            (ROOT / "config" / "rl_baseline.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["algorithm"]["name"], "MaskablePPO")
        self.assertEqual(config["training"]["default_representation"], "ubr-1.2")
        self.assertEqual(config["research_policy"]["universal_representation"], "ubr-1.2")
        self.assertEqual(config["research_policy"]["observation_dim"], 518)
        self.assertEqual(config["research_policy"]["action_dim"], 10)
        self.assertTrue(config["evaluation"]["require_same_generation"])

    def test_multiseed_defaults(self):
        config = yaml.safe_load(
            (ROOT / "config" / "rl_baseline.yaml").read_text(encoding="utf-8")
        )
        block = config["multiseed"]
        self.assertEqual(block["representations"], ["ubr-1.1", "ubr-1.2"])
        self.assertEqual(block["seeds"], [42, 123, 456, 789, 2026])
        self.assertEqual(block["timesteps"], 10000)
        self.assertEqual(block["evaluation_opponents"], ["random", "type_aware"])
        self.assertEqual(block["repeats_per_leg"], 10)


    def test_budget_study_defaults(self):
        config = yaml.safe_load(
            (ROOT / "config" / "rl_baseline.yaml").read_text(encoding="utf-8")
        )
        block = config["budget_study"]
        self.assertEqual(block["representations"], ["ubr-1.1", "ubr-1.2"])
        self.assertEqual(block["seeds"], [42, 123, 456, 789, 2026])
        self.assertEqual(block["budgets"], [10000, 50000])
        self.assertEqual(block["baseline_budget"], 10000)
        self.assertEqual(block["evaluation_opponents"], ["random", "type_aware"])
        self.assertEqual(block["repeats_per_leg"], 10)


if __name__ == "__main__":
    unittest.main()
