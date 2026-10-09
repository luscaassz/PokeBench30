import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class RLConfigTests(unittest.TestCase):
    def test_rl_baseline_contract(self):
        config = yaml.safe_load(
            (ROOT / "config" / "rl_baseline.yaml").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            config["algorithm"]["name"],
            "MaskablePPO",
        )
        self.assertEqual(
            config["research_policy"]["universal_representation"],
            "ubr-1.1",
        )
        self.assertEqual(
            config["research_policy"]["observation_dim"],
            472,
        )
        self.assertEqual(
            config["research_policy"]["action_dim"],
            10,
        )
        self.assertTrue(
            config["evaluation"]["require_same_generation"]
        )


if __name__ == "__main__":
    unittest.main()
