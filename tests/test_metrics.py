import unittest

from pokebench30.evaluation.metrics import exact_binomial_two_sided, wilson_interval


class MetricsTests(unittest.TestCase):
    def test_no_trials(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_interval_contains_half(self):
        low, high = wilson_interval(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_binomial_balanced(self):
        self.assertAlmostEqual(exact_binomial_two_sided(5, 10), 1.0)

    def test_binomial_extreme(self):
        self.assertLess(exact_binomial_two_sided(10, 10), 0.01)


if __name__ == "__main__":
    unittest.main()
