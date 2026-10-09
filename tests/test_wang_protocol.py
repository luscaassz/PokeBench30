import math
import unittest

from pokebench30.rl.wang_protocol import (
    WANG_INITIAL_LR,
    WangLearningRateSchedule,
    WangPPOConfig,
)


class WangProtocolTests(unittest.TestCase):
    def test_initial_lr(self):
        self.assertAlmostEqual(
            WANG_INITIAL_LR,
            10 ** (-4.23),
            places=14,
        )

    def test_lr_schedule_endpoints(self):
        schedule = WangLearningRateSchedule()
        self.assertAlmostEqual(
            schedule(1.0),
            WANG_INITIAL_LR,
            places=14,
        )
        expected_final = WANG_INITIAL_LR / (9.0 ** 1.5)
        self.assertAlmostEqual(
            schedule(0.0),
            expected_final,
            places=14,
        )

    def test_lr_schedule_monotonic(self):
        schedule = WangLearningRateSchedule()
        values = [
            schedule(progress)
            for progress in (1.0, 0.75, 0.5, 0.25, 0.0)
        ]
        self.assertTrue(
            all(left >= right for left, right in zip(values, values[1:]))
        )

    def test_published_hyperparameters(self):
        cfg = WangPPOConfig()
        self.assertEqual(cfg.n_epochs, 7)
        self.assertAlmostEqual(cfg.gamma, 0.9999)
        self.assertAlmostEqual(cfg.gae_lambda, 0.754)
        self.assertAlmostEqual(cfg.clip_range, 0.0829)
        self.assertAlmostEqual(cfg.clip_range_vf, 0.0184)
        self.assertAlmostEqual(cfg.ent_coef, 0.0588)
        self.assertAlmostEqual(cfg.vf_coef, 0.4375)
        self.assertAlmostEqual(cfg.max_grad_norm, 0.5430)


if __name__ == "__main__":
    unittest.main()
