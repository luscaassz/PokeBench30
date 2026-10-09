from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Masked DQN requires PyTorch. Run: python -m pip install -r requirements.txt"
    ) from exc


NEGATIVE_MASK_VALUE = -1.0e9


@dataclass(frozen=True)
class DQNConfig:
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 50_000
    learning_starts: int = 5_000
    train_frequency: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 2_000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_fraction: float = 0.30
    max_grad_norm: float = 10.0
    hidden_sizes: tuple[int, ...] = (256, 128)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int] = (256, 128),
    ) -> None:
        super().__init__()
        dims = [int(input_dim), *[int(v) for v in hidden_sizes], int(action_dim)]
        layers: list[nn.Module] = []
        for left, right in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(left, right))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class ReplayBuffer:
    """Simple ring buffer with strict next-state action masks."""

    def __init__(
        self,
        *,
        capacity: int,
        observation_dim: int,
        action_dim: int,
    ) -> None:
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

        if self.capacity <= 0:
            raise ValueError("Replay capacity must be > 0.")

        self.observations = np.zeros(
            (self.capacity, self.observation_dim), dtype=np.float32
        )
        self.next_observations = np.zeros(
            (self.capacity, self.observation_dim), dtype=np.float32
        )
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.next_masks = np.ones(
            (self.capacity, self.action_dim), dtype=np.uint8
        )

        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return int(self.size)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        idx = int(self.position)
        self.observations[idx] = np.asarray(observation, dtype=np.float32)
        self.next_observations[idx] = np.asarray(
            next_observation, dtype=np.float32
        )
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = float(bool(done))

        mask = np.asarray(next_action_mask, dtype=bool).reshape(-1)
        if mask.shape != (self.action_dim,):
            raise ValueError(
                f"Expected next action mask {(self.action_dim,)}, got {mask.shape}."
            )
        # Terminal transitions do not bootstrap, but retain a nonempty mask to
        # keep tensor operations finite and easy to validate.
        if bool(done):
            mask = np.ones(self.action_dim, dtype=bool)
        if not np.any(mask):
            raise ValueError("Nonterminal transition has an empty next-action mask.")
        self.next_masks[idx] = mask.astype(np.uint8)

        self.position = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        if self.size < int(batch_size):
            raise ValueError(
                f"Replay buffer has {self.size} transitions, "
                f"cannot sample batch_size={batch_size}."
            )
        indices = rng.integers(0, self.size, size=int(batch_size))
        return {
            "observations": self.observations[indices],
            "next_observations": self.next_observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "next_masks": self.next_masks[indices].astype(bool),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            capacity=np.int64(self.capacity),
            observation_dim=np.int64(self.observation_dim),
            action_dim=np.int64(self.action_dim),
            position=np.int64(self.position),
            size=np.int64(self.size),
            observations=self.observations[: self.size],
            next_observations=self.next_observations[: self.size],
            actions=self.actions[: self.size],
            rewards=self.rewards[: self.size],
            dones=self.dones[: self.size],
            next_masks=self.next_masks[: self.size],
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            capacity = int(data["capacity"])
            observation_dim = int(data["observation_dim"])
            action_dim = int(data["action_dim"])
            buffer = cls(
                capacity=capacity,
                observation_dim=observation_dim,
                action_dim=action_dim,
            )
            size = int(data["size"])
            position = int(data["position"])
            if not 0 <= size <= capacity:
                raise ValueError("Invalid replay-buffer size.")
            if not 0 <= position < capacity:
                raise ValueError("Invalid replay-buffer position.")

            if size:
                buffer.observations[:size] = data["observations"]
                buffer.next_observations[:size] = data["next_observations"]
                buffer.actions[:size] = data["actions"]
                buffer.rewards[:size] = data["rewards"]
                buffer.dones[:size] = data["dones"]
                buffer.next_masks[:size] = data["next_masks"]

            buffer.size = size
            buffer.position = position
            return buffer


class MaskedDQNTrainer:
    """DQN / Double DQN with legal-action masking in behavior and targets."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        algorithm: str,
        config: DQNConfig,
        total_environment_steps: int,
        seed: int,
        device: str = "auto",
    ) -> None:
        if algorithm not in {"MaskedDQN", "MaskedDoubleDQN"}:
            raise ValueError(
                "algorithm must be 'MaskedDQN' or 'MaskedDoubleDQN'."
            )

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.algorithm = algorithm
        self.config = config
        self.total_environment_steps = int(total_environment_steps)
        self.seed = int(seed)

        if device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = str(device)
        self.device = torch.device(resolved)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.rng = np.random.default_rng(self.seed)

        self.online_network = QNetwork(
            self.observation_dim,
            self.action_dim,
            self.config.hidden_sizes,
        ).to(self.device)
        self.target_network = QNetwork(
            self.observation_dim,
            self.action_dim,
            self.config.hidden_sizes,
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=float(self.config.learning_rate),
        )

        self.replay = ReplayBuffer(
            capacity=self.config.replay_capacity,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
        )

        self.environment_steps = 0
        self.gradient_updates = 0

    def parameter_count(self) -> int:
        return int(
            sum(parameter.numel() for parameter in self.online_network.parameters())
        )

    def epsilon(self, step: int | None = None) -> float:
        current = int(self.environment_steps if step is None else step)
        duration = max(
            1,
            int(
                round(
                    self.total_environment_steps
                    * float(self.config.epsilon_fraction)
                )
            ),
        )
        fraction = min(max(current / duration, 0.0), 1.0)
        return float(
            self.config.epsilon_start
            + fraction
            * (self.config.epsilon_final - self.config.epsilon_start)
        )

    @staticmethod
    def _validate_mask(mask: np.ndarray, action_dim: int) -> np.ndarray:
        value = np.asarray(mask, dtype=bool).reshape(-1)
        if value.shape != (int(action_dim),):
            raise ValueError(
                f"Expected action mask {(int(action_dim),)}, got {value.shape}."
            )
        if not np.any(value):
            raise ValueError("Action mask contains no legal actions.")
        return value

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> int:
        mask = self._validate_mask(action_mask, self.action_dim)
        legal = np.flatnonzero(mask)

        if not deterministic and self.rng.random() < self.epsilon():
            return int(self.rng.choice(legal))

        obs = torch.as_tensor(
            np.asarray(observation, dtype=np.float32).reshape(1, -1),
            device=self.device,
        )
        with torch.no_grad():
            q_values = self.online_network(obs)[0]
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            q_values = q_values.masked_fill(~mask_tensor, NEGATIVE_MASK_VALUE)
            return int(torch.argmax(q_values).item())

    def observe(
        self,
        *,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        self.replay.add(
            observation,
            action,
            reward,
            next_observation,
            done,
            next_action_mask,
        )
        self.environment_steps += 1

    def _masked_dqn_target(
        self,
        next_observations: torch.Tensor,
        next_masks: torch.Tensor,
    ) -> torch.Tensor:
        target_q = self.target_network(next_observations)
        target_q = target_q.masked_fill(
            ~next_masks,
            NEGATIVE_MASK_VALUE,
        )
        return target_q.max(dim=1).values

    def _masked_double_dqn_target(
        self,
        next_observations: torch.Tensor,
        next_masks: torch.Tensor,
    ) -> torch.Tensor:
        online_q = self.online_network(next_observations)
        online_q = online_q.masked_fill(
            ~next_masks,
            NEGATIVE_MASK_VALUE,
        )
        next_actions = online_q.argmax(dim=1, keepdim=True)

        target_q = self.target_network(next_observations)
        return target_q.gather(1, next_actions).squeeze(1)

    def gradient_step(self) -> float | None:
        if self.environment_steps < int(self.config.learning_starts):
            return None
        if len(self.replay) < int(self.config.batch_size):
            return None

        batch = self.replay.sample(self.config.batch_size, self.rng)

        observations = torch.as_tensor(
            batch["observations"], dtype=torch.float32, device=self.device
        )
        next_observations = torch.as_tensor(
            batch["next_observations"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.int64, device=self.device
        )
        rewards = torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            batch["dones"], dtype=torch.float32, device=self.device
        )
        next_masks = torch.as_tensor(
            batch["next_masks"], dtype=torch.bool, device=self.device
        )

        q_values = self.online_network(observations)
        chosen_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.algorithm == "MaskedDoubleDQN":
                next_q = self._masked_double_dqn_target(
                    next_observations,
                    next_masks,
                )
            else:
                next_q = self._masked_dqn_target(
                    next_observations,
                    next_masks,
                )
            targets = rewards + (
                float(self.config.gamma) * (1.0 - dones) * next_q
            )

        loss = F.smooth_l1_loss(chosen_q, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            float(self.config.max_grad_norm),
        )
        self.optimizer.step()
        self.gradient_updates += 1

        return float(loss.detach().cpu().item())

    def maybe_train(self) -> list[float]:
        if self.environment_steps < int(self.config.learning_starts):
            return []
        if self.environment_steps % int(self.config.train_frequency) != 0:
            return []

        losses: list[float] = []
        for _ in range(int(self.config.gradient_steps)):
            loss = self.gradient_step()
            if loss is not None:
                losses.append(loss)

        if (
            self.environment_steps > 0
            and self.environment_steps % int(self.config.target_update_interval) == 0
        ):
            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

        return losses

    def checkpoint_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": "pokebench30_masked_dqn_v1",
            "algorithm": self.algorithm,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "config": self.config.to_dict(),
            "total_environment_steps": self.total_environment_steps,
            "seed": self.seed,
            "environment_steps": self.environment_steps,
            "gradient_updates": self.gradient_updates,
            "online_state_dict": self.online_network.state_dict(),
            "target_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "numpy_rng_state_json": json.dumps(self.rng.bit_generator.state),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return payload

    def save_model(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), path)
        return path

    def save_resume_state(self, directory: str | Path, nominal_checkpoint: int) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        model_path = directory / "trainer_state.pt"
        replay_path = directory / "replay_buffer.npz"
        marker_path = directory / "resume_metadata.json"

        self.save_model(model_path)
        self.replay.save(replay_path)
        marker_path.write_text(
            json.dumps(
                {
                    "nominal_checkpoint": int(nominal_checkpoint),
                    "environment_steps": int(self.environment_steps),
                    "algorithm": self.algorithm,
                    "replay_size": len(self.replay),
                    "replay_capacity": self.replay.capacity,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load_training_state(
        cls,
        *,
        model_path: str | Path,
        replay_path: str | Path,
        device: str = "auto",
    ) -> "MaskedDQNTrainer":
        payload = torch.load(
            Path(model_path),
            map_location="cpu",
            weights_only=False,
        )
        config_raw = dict(payload["config"])
        config_raw["hidden_sizes"] = tuple(config_raw["hidden_sizes"])
        config = DQNConfig(**config_raw)

        trainer = cls(
            observation_dim=int(payload["observation_dim"]),
            action_dim=int(payload["action_dim"]),
            algorithm=str(payload["algorithm"]),
            config=config,
            total_environment_steps=int(payload["total_environment_steps"]),
            seed=int(payload["seed"]),
            device=device,
        )
        trainer.online_network.load_state_dict(payload["online_state_dict"])
        trainer.target_network.load_state_dict(payload["target_state_dict"])
        trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
        trainer.environment_steps = int(payload["environment_steps"])
        trainer.gradient_updates = int(payload["gradient_updates"])
        trainer.replay = ReplayBuffer.load(replay_path)

        trainer.rng.bit_generator.state = json.loads(
            payload["numpy_rng_state_json"]
        )
        torch.set_rng_state(payload["torch_rng_state"])
        if (
            torch.cuda.is_available()
            and "cuda_rng_state_all" in payload
        ):
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])

        return trainer


class MaskedQPolicy:
    """Inference-only masked Q-network."""

    def __init__(
        self,
        *,
        network: QNetwork,
        algorithm: str,
        observation_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.network = network
        self.algorithm = algorithm
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.network.eval()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
    ) -> "MaskedQPolicy":
        if device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = str(device)
        torch_device = torch.device(resolved)

        payload = torch.load(
            Path(path),
            map_location=torch_device,
            weights_only=False,
        )
        config = payload["config"]
        network = QNetwork(
            int(payload["observation_dim"]),
            int(payload["action_dim"]),
            tuple(config["hidden_sizes"]),
        ).to(torch_device)
        network.load_state_dict(payload["online_state_dict"])
        return cls(
            network=network,
            algorithm=str(payload["algorithm"]),
            observation_dim=int(payload["observation_dim"]),
            action_dim=int(payload["action_dim"]),
            device=torch_device,
        )

    def predict(
        self,
        observation: np.ndarray,
        *,
        action_mask: np.ndarray,
    ) -> int:
        mask = MaskedDQNTrainer._validate_mask(
            action_mask,
            self.action_dim,
        )
        obs = torch.as_tensor(
            np.asarray(observation, dtype=np.float32).reshape(1, -1),
            device=self.device,
        )
        with torch.no_grad():
            q_values = self.network(obs)[0]
            mask_tensor = torch.as_tensor(
                mask, dtype=torch.bool, device=self.device
            )
            q_values = q_values.masked_fill(
                ~mask_tensor,
                NEGATIVE_MASK_VALUE,
            )
            return int(torch.argmax(q_values).item())
