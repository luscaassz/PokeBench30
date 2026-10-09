from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.spaces import Box
    import torch as th
    from torch import nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Temporal policy architectures require gymnasium, torch and "
        "stable-baselines3. Run: python -m pip install -r requirements.txt"
    ) from exc

from .policy_specs import PolicySpec, padded_history


class TemporalStackAdapter(gym.Wrapper):
    """Expose a fixed window of recent UBR observations to MaskablePPO.

    The underlying action mask always remains the *current* state's mask.
    Older masks are intentionally not included in the observation.
    """

    def __init__(self, env: gym.Env, *, history_len: int):
        super().__init__(env)
        if history_len <= 1:
            raise ValueError("TemporalStackAdapter requires history_len > 1.")

        if not isinstance(env.observation_space, Box):
            raise TypeError("TemporalStackAdapter requires a Box observation space.")

        self.history_len = int(history_len)
        self.base_dim = int(np.prod(env.observation_space.shape))
        self._history: deque[np.ndarray] = deque(maxlen=self.history_len)

        low = np.tile(
            np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1),
            self.history_len,
        )
        high = np.tile(
            np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1),
            self.history_len,
        )
        self.observation_space = Box(
            low=low,
            high=high,
            shape=(self.base_dim * self.history_len,),
            dtype=np.float32,
        )
        self.action_space = env.action_space

    def _stack(self, obs: Any) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.shape != (self.base_dim,):
            raise ValueError(
                f"Temporal observation shape mismatch: "
                f"expected {(self.base_dim,)}, got {arr.shape}."
            )
        self._history.append(arr.copy())
        return padded_history(
            self._history,
            history_len=self.history_len,
            base_dim=self.base_dim,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._history.clear()
        return self._stack(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._stack(obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.env.action_masks(), dtype=bool)


class TemporalLSTMExtractor(BaseFeaturesExtractor):
    """Encode a fixed observation window with an LSTM.

    This is intentionally a fixed-window temporal encoder rather than a
    stateful RecurrentPPO policy. It preserves MaskablePPO's strict action
    masking while allowing the architecture study to test sequence modeling.
    """

    def __init__(
        self,
        observation_space: Box,
        *,
        base_dim: int,
        history_len: int,
        hidden_size: int = 256,
        num_layers: int = 1,
    ):
        self.base_dim = int(base_dim)
        self.history_len = int(history_len)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        expected = self.base_dim * self.history_len
        actual = int(np.prod(observation_space.shape))
        if actual != expected:
            raise ValueError(
                f"TemporalLSTMExtractor expected input dim {expected}, got {actual}."
            )

        super().__init__(observation_space, features_dim=self.hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.base_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch = observations.shape[0]
        sequence = observations.reshape(
            batch,
            self.history_len,
            self.base_dim,
        )
        _, (hidden, _) = self.lstm(sequence)
        return hidden[-1]


def wrap_env_for_policy(env: gym.Env, spec: PolicySpec) -> gym.Env:
    if spec.history_len <= 1:
        return env
    return TemporalStackAdapter(env, history_len=spec.history_len)


def policy_kwargs_for_spec(spec: PolicySpec, *, base_dim: int) -> dict[str, Any]:
    if spec.family == "history_lstm":
        return {
            "features_extractor_class": TemporalLSTMExtractor,
            "features_extractor_kwargs": {
                "base_dim": int(base_dim),
                "history_len": int(spec.history_len),
                "hidden_size": int(spec.temporal_hidden_size or 256),
                "num_layers": int(spec.temporal_layers),
            },
            "net_arch": list(spec.net_arch),
        }

    return {"net_arch": list(spec.net_arch)}
