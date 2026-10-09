from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Wang-style PPO requires torch and stable-baselines3. "
        "Run: python -m pip install -r requirements.txt"
    ) from exc

from poke_env.battle import AbstractBattle

from .env import UniversalRLSinglesEnv
from .wang_protocol import (
    WANG_INITIAL_LR,
    WangLearningRateSchedule,
    WangPPOConfig,
)


class SparseTerminalSelfPlayEnv(UniversalRLSinglesEnv):
    """Universal PettingZoo singles env with Wang's sparse terminal reward.

    Used directly as a two-player PettingZoo environment. After conversion to
    a Stable-Baselines3 vector environment, both player perspectives are fed to
    the same PPO policy, so both trajectories contribute to training.
    """

    def calc_reward(self, battle: AbstractBattle) -> float:
        # Reuse poke-env's battle-value difference machinery, but zero every
        # shaping term except battle outcome. This yields the desired
        # terminal reward profile: +1 win, -1 loss, 0 otherwise.
        return self.reward_computing_helper(
            battle,
            fainted_value=0.0,
            hp_value=0.0,
            status_value=0.0,
            victory_value=1.0,
        )


class UniversalObservationExtractor(BaseFeaturesExtractor):
    """Extract UBR observations while ignoring the transport action mask."""

    def __init__(
        self,
        observation_space,
        *,
        hidden_dim: int = 256,
        trunk_layers: int = 3,
    ) -> None:
        base_space = observation_space["observation"]
        input_dim = int(np.prod(base_space.shape))
        hidden_dim = int(hidden_dim)
        trunk_layers = int(trunk_layers)

        if trunk_layers <= 0:
            raise ValueError("trunk_layers must be > 0")

        super().__init__(
            observation_space,
            features_dim=hidden_dim,
        )

        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(trunk_layers):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(nn.ReLU())
            current = hidden_dim
        self.network = nn.Sequential(*layers)

    def forward(self, observations):
        values = observations["observation"]

        # Defense in depth. The UBR encoder is finite-safe, but keep the
        # policy boundary robust to any third-party vector-wrapper anomaly.
        if not torch.isfinite(values).all():
            values = torch.nan_to_num(
                values,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )

        return self.network(values)


class WangMaskedActorCriticPolicy(ActorCriticPolicy):
    """PPO actor-critic with invalid actions masked at the logits.

    This follows the masking pattern used in poke-env's official self-play
    example and matches Wang's methodological requirement that invalid actions
    be removed before the final softmax.

    The observation dict contains:
        observation: UBR vector
        action_mask: legal-action binary mask
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        hidden_dim = int(kwargs.pop("wang_hidden_dim", 256))
        trunk_layers = int(kwargs.pop("wang_trunk_layers", 3))
        head_layers = int(kwargs.pop("wang_head_layers", 2))

        head = [hidden_dim] * head_layers

        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=UniversalObservationExtractor,
            features_extractor_kwargs={
                "hidden_dim": hidden_dim,
                "trunk_layers": trunk_layers,
            },
            net_arch={
                "pi": head,
                "vf": head,
            },
            activation_fn=nn.ReLU,
            share_features_extractor=True,
        )

        self._pokebench_action_mask = None

    def _set_action_mask(self, obs) -> None:
        mask = obs["action_mask"]
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, device=self.device)
        self._pokebench_action_mask = mask.to(dtype=torch.bool)

    def forward(self, obs, deterministic: bool = False):
        self._set_action_mask(obs)
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._set_action_mask(obs)
        return super().evaluate_actions(obs, actions)

    def get_distribution(self, obs):
        self._set_action_mask(obs)
        return super().get_distribution(obs)

    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)

        if self._pokebench_action_mask is None:
            raise RuntimeError(
                "Action distribution requested before an action mask was set."
            )

        mask = self._pokebench_action_mask
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)

        if mask.shape != action_logits.shape:
            try:
                mask = mask.reshape(action_logits.shape)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Action-mask/logit shape mismatch: "
                    f"mask={tuple(mask.shape)}, "
                    f"logits={tuple(action_logits.shape)}"
                ) from exc

        masked_logits = action_logits.masked_fill(
            ~mask,
            float("-inf"),
        )

        return self.action_dist.proba_distribution(masked_logits)


def wang_policy_kwargs(config: WangPPOConfig) -> dict[str, Any]:
    return {
        "wang_hidden_dim": int(config.hidden_dim),
        "wang_trunk_layers": int(config.trunk_layers),
        "wang_head_layers": int(config.head_layers),
    }
