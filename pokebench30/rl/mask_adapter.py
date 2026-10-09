from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.spaces import Dict as DictSpace
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PokéBench30 MaskablePPO adapter requires gymnasium. "
        "Run: python -m pip install -r requirements.txt"
    ) from exc


class SB3MaskAdapter(gym.Wrapper):
    """Convert poke-env's observation+mask dict into MaskablePPO's interface.

    poke-env exposes:
        {
            "observation": float vector,
            "action_mask": binary vector,
        }

    sb3-contrib MaskablePPO expects:
        observation -> normal Gym observation
        env.action_masks() -> bool vector
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        original_space = env.observation_space
        if (
            isinstance(original_space, DictSpace)
            and "observation" in original_space.spaces
        ):
            self.observation_space = original_space["observation"]
        else:
            self.observation_space = original_space

        self.action_space = env.action_space
        self._last_action_mask: np.ndarray | None = None

    def _unpack(self, obs: Any):
        if isinstance(obs, dict) and "observation" in obs:
            mask = obs.get("action_mask")
            if mask is not None:
                self._last_action_mask = np.asarray(
                    mask,
                    dtype=bool,
                )
            return np.asarray(
                obs["observation"],
                dtype=np.float32,
            )

        return np.asarray(obs, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._unpack(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._unpack(obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def action_masks(self) -> np.ndarray:
        if self._last_action_mask is None:
            raise RuntimeError(
                "Action mask requested before the environment produced "
                "its first observation. Call reset() first."
            )

        if not np.any(self._last_action_mask):
            raise RuntimeError(
                "Environment produced an all-zero action mask."
            )

        return self._last_action_mask.copy()
