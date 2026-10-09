import unittest
from pathlib import Path

import yaml

from pokebench30.rl.crossgen_study import (
    actual_rollout_checkpoint,
    expected_actual_final,
    nominal_checkpoints,
    rollout_transition_block,
)


ROOT = Path(__file__).resolve().parents[1]


class CrossGenerationStudyTests(unittest.TestCase):
    def test_nominal_checkpoints(self):
        self.assertEqual(
            nominal_checkpoints(900000, 150000),
            [150000, 300000, 450000, 600000, 750000, 900000],
        )

    def test_rollout_rounding(self):
        ppo = {
            "n_steps": 2048,
            "parallel_env_copies": 1,
        }
        block = rollout_transition_block(ppo)
        self.assertEqual(block, 4096)
        self.assertEqual(
            actual_rollout_checkpoint(150000, block),
            151552,
        )
        self.assertEqual(
            expected_actual_final(900000, block),
            901120,
        )

    def test_fixed_pool_has_ten_pairs_per_generation(self):
        config = yaml.safe_load(
            (ROOT / "config" / "fixed_teams.yaml").read_text(
                encoding="utf-8"
            )
        )
        for generation in (1, 4, 9):
            pairs = config["generations"][generation]["pairs"]
            self.assertEqual(len(pairs), 10)
            self.assertEqual(len({row["id"] for row in pairs}), 10)

    def test_crossgen_config_is_frozen_to_target_generations(self):
        config = yaml.safe_load(
            (ROOT / "config" / "crossgen_benchmark.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["study"]["generations"], [1, 4, 9])
        self.assertEqual(
            config["study"]["seeds"],
            [42, 123, 456, 789, 2026],
        )
        self.assertEqual(config["study"]["total_timesteps"], 900000)
        self.assertEqual(
            config["study"]["fixed_team_pairs_per_generation"],
            10,
        )


if __name__ == "__main__":
    unittest.main()
