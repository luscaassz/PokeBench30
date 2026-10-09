import unittest

import numpy as np

from pokebench30.rl.policy_specs import (
    DEFAULT_POLICY_ARCHITECTURES,
    get_policy_spec,
    padded_history,
    validate_architectures,
)


class PolicySpecsTests(unittest.TestCase):
    def test_default_architectures(self):
        self.assertEqual(
            DEFAULT_POLICY_ARCHITECTURES,
            ["mlp_baseline", "mlp_large", "history_mlp", "history_lstm"],
        )

    def test_model_input_dims_for_ubr12(self):
        self.assertEqual(get_policy_spec("mlp_baseline").model_input_dim(518), 518)
        self.assertEqual(get_policy_spec("mlp_large").model_input_dim(518), 518)
        self.assertEqual(get_policy_spec("history_mlp").model_input_dim(518), 2072)
        self.assertEqual(get_policy_spec("history_lstm").model_input_dim(518), 2072)

    def test_history_is_zero_left_padded(self):
        a = np.ones(3, dtype=np.float32)
        b = np.full(3, 2.0, dtype=np.float32)
        stacked = padded_history([a, b], history_len=4, base_dim=3)
        self.assertEqual(stacked.shape, (12,))
        np.testing.assert_allclose(stacked[:6], 0.0)
        np.testing.assert_allclose(stacked[6:9], 1.0)
        np.testing.assert_allclose(stacked[9:12], 2.0)

    def test_history_keeps_most_recent(self):
        vectors = [
            np.full(2, float(i), dtype=np.float32)
            for i in range(1, 6)
        ]
        stacked = padded_history(vectors, history_len=4, base_dim=2)
        np.testing.assert_allclose(
            stacked,
            np.array([2, 2, 3, 3, 4, 4, 5, 5], dtype=np.float32),
        )

    def test_unknown_architecture_rejected(self):
        with self.assertRaises(ValueError):
            validate_architectures(["does_not_exist"])


if __name__ == "__main__":
    unittest.main()
