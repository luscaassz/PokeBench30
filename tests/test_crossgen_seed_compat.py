import unittest

from pokebench30.rl.crossgen_study import _patch_supersuit_seed_compat


class DummyBase:
    def __init__(self):
        self.num_envs = 2
        self.calls = []

    def reset(self, *, seed=None, options=None):
        self.calls.append((seed, options))
        return "obs", []


class DummyWrapper:
    def __init__(self):
        self.venv = DummyBase()
        self.num_envs = 2


class SeedCompatTests(unittest.TestCase):
    def test_seed_is_consumed_by_next_reset(self):
        env = DummyWrapper()
        _patch_supersuit_seed_compat(env)

        returned = env.venv.seed(42)
        self.assertEqual(returned, [42, 43])
        self.assertEqual(env.venv.calls, [])

        env.venv.reset()
        self.assertEqual(env.venv.calls, [(42, None)])

        env.venv.reset()
        self.assertEqual(env.venv.calls[-1], (None, None))

    def test_existing_seed_method_is_left_untouched(self):
        class AlreadyCompatible(DummyBase):
            def seed(self, value=None):
                return ["original", value]

        env = DummyWrapper()
        env.venv = AlreadyCompatible()
        _patch_supersuit_seed_compat(env)

        self.assertEqual(env.venv.seed(7), ["original", 7])


if __name__ == "__main__":
    unittest.main()
