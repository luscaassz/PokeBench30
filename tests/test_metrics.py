import unittest

from pokebench30.evaluation.metrics import wilson_interval


class WilsonIntervalTests(unittest.TestCase):
    def test_no_trials(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_interval_contains_half_for_balanced_sample(self):
        low, high = wilson_interval(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_bounds_are_valid(self):
        low, high = wilson_interval(10, 10)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLessEqual(low, high)


if __name__ == "__main__":
    unittest.main()
