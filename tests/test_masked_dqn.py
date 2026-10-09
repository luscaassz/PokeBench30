import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pokebench30.rl.masked_dqn import (
    DQNConfig,
    MaskedDQNTrainer,
    MaskedQPolicy,
    ReplayBuffer,
)


class MaskedDQNTests(unittest.TestCase):
    def make_trainer(self, algorithm="MaskedDQN"):
        config = DQNConfig(
            learning_rate=1e-3,
            gamma=0.99,
            batch_size=2,
            replay_capacity=8,
            learning_starts=0,
            train_frequency=1,
            gradient_steps=1,
            target_update_interval=2,
            epsilon_start=0.0,
            epsilon_final=0.0,
            epsilon_fraction=1.0,
            hidden_sizes=(),
        )
        return MaskedDQNTrainer(
            observation_dim=2,
            action_dim=3,
            algorithm=algorithm,
            config=config,
            total_environment_steps=10,
            seed=42,
            device="cpu",
        )

    def test_behavior_mask_blocks_illegal_argmax(self):
        trainer = self.make_trainer()
        layer = trainer.online_network.network[-1]
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.copy_(torch.tensor([1.0, 100.0, 2.0]))

        action = trainer.select_action(
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([1, 0, 1], dtype=bool),
            deterministic=True,
        )
        self.assertEqual(action, 2)

    def test_dqn_target_masks_illegal_actions(self):
        trainer = self.make_trainer("MaskedDQN")
        layer = trainer.target_network.network[-1]
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.copy_(torch.tensor([1.0, 100.0, 3.0]))

        obs = torch.zeros((1, 2), dtype=torch.float32)
        mask = torch.tensor([[True, False, True]])
        target = trainer._masked_dqn_target(obs, mask)
        self.assertAlmostEqual(float(target.item()), 3.0, places=6)

    def test_double_dqn_uses_online_selection_target_evaluation(self):
        trainer = self.make_trainer("MaskedDoubleDQN")
        online = trainer.online_network.network[-1]
        target = trainer.target_network.network[-1]

        with torch.no_grad():
            online.weight.zero_()
            online.bias.copy_(torch.tensor([5.0, 100.0, 7.0]))
            target.weight.zero_()
            target.bias.copy_(torch.tensor([11.0, 99.0, 13.0]))

        obs = torch.zeros((1, 2), dtype=torch.float32)
        mask = torch.tensor([[True, False, True]])
        value = trainer._masked_double_dqn_target(obs, mask)

        # Online chooses legal action 2; target evaluates action 2 as 13.
        self.assertAlmostEqual(float(value.item()), 13.0, places=6)

    def test_replay_roundtrip(self):
        buffer = ReplayBuffer(
            capacity=4,
            observation_dim=2,
            action_dim=3,
        )
        buffer.add(
            np.array([1.0, 2.0], dtype=np.float32),
            1,
            3.5,
            np.array([4.0, 5.0], dtype=np.float32),
            False,
            np.array([1, 0, 1], dtype=bool),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.npz"
            buffer.save(path)
            restored = ReplayBuffer.load(path)

        self.assertEqual(len(restored), 1)
        np.testing.assert_allclose(restored.observations[0], [1.0, 2.0])
        self.assertEqual(int(restored.actions[0]), 1)
        self.assertEqual(restored.next_masks[0].tolist(), [1, 0, 1])

    def test_checkpoint_inference_roundtrip(self):
        trainer = self.make_trainer("MaskedDQN")
        layer = trainer.online_network.network[-1]
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.copy_(torch.tensor([2.0, 9.0, 4.0]))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "masked_dqn.pt"
            trainer.save_model(path)
            policy = MaskedQPolicy.load(path, device="cpu")
            action = policy.predict(
                np.array([0.0, 0.0], dtype=np.float32),
                action_mask=np.array([1, 0, 1], dtype=bool),
            )

        self.assertEqual(action, 2)


if __name__ == "__main__":
    unittest.main()
