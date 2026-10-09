import unittest

from pokebench30.rl.algorithm_study import (
    Stage,
    effective_segment_steps,
    nominal_checkpoints,
    opponent_at,
    validate_curriculum,
)


class AlgorithmStudyTests(unittest.TestCase):
    def setUp(self):
        self.stages = [
            Stage("random", 100000),
            Stage("type_aware", 200000),
        ]

    def test_curriculum(self):
        self.assertEqual(opponent_at(self.stages, 0), "random")
        self.assertEqual(opponent_at(self.stages, 99999), "random")
        self.assertEqual(opponent_at(self.stages, 100000), "type_aware")
        self.assertEqual(opponent_at(self.stages, 299999), "type_aware")

    def test_nominal_checkpoints(self):
        self.assertEqual(
            nominal_checkpoints(300000, 50000),
            [50000, 100000, 150000, 200000, 250000, 300000],
        )

    def test_effective_steps_match_ppo_rollout_rounding(self):
        self.assertEqual(effective_segment_steps(50000, 2048), 51200)
        self.assertEqual(
            6 * effective_segment_steps(50000, 2048),
            307200,
        )

    def test_curriculum_validation(self):
        validate_curriculum(
            self.stages,
            total_timesteps=300000,
            checkpoint_interval=50000,
        )

    def test_invalid_total_rejected(self):
        with self.assertRaises(ValueError):
            validate_curriculum(
                self.stages,
                total_timesteps=250000,
                checkpoint_interval=50000,
            )


if __name__ == "__main__":
    unittest.main()
