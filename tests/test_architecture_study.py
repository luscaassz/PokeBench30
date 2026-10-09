import unittest

from pokebench30.rl.architecture_study import (
    Stage,
    checkpoints,
    opponent_at,
    validate_curriculum,
)


class ArchitectureStudyTests(unittest.TestCase):
    def setUp(self):
        self.stages = [
            Stage("random", 100000),
            Stage("type_aware", 200000),
        ]

    def test_curriculum_schedule(self):
        self.assertEqual(opponent_at(self.stages, 0), "random")
        self.assertEqual(opponent_at(self.stages, 99999), "random")
        self.assertEqual(opponent_at(self.stages, 100000), "type_aware")
        self.assertEqual(opponent_at(self.stages, 299999), "type_aware")

    def test_checkpoints(self):
        self.assertEqual(
            checkpoints(300000, 50000),
            [50000, 100000, 150000, 200000, 250000, 300000],
        )

    def test_curriculum_validation(self):
        validate_curriculum(
            self.stages,
            total_timesteps=300000,
            checkpoint_interval=50000,
        )

    def test_misaligned_transition_rejected(self):
        with self.assertRaises(ValueError):
            validate_curriculum(
                [Stage("random", 75000), Stage("type_aware", 225000)],
                total_timesteps=300000,
                checkpoint_interval=50000,
            )


if __name__ == "__main__":
    unittest.main()
