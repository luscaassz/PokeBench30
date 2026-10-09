from __future__ import annotations

from dataclasses import dataclass
from typing import Any


WANG_INITIAL_LR = 10 ** (-4.23)


@dataclass(frozen=True)
class WangPPOConfig:
    """Published PPO optimization settings adapted for PokéBench30 v0.6."""

    n_epochs: int = 7
    gamma: float = 0.9999
    gae_lambda: float = 0.754
    clip_range: float = 0.0829
    clip_range_vf: float = 0.0184
    ent_coef: float = 0.0588
    vf_coef: float = 0.4375
    max_grad_norm: float = 0.5430

    # Adapted for a single-environment portable implementation.
    n_steps: int = 4096
    batch_size: int = 1024

    hidden_dim: int = 256
    trunk_layers: int = 3
    head_layers: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate_initial": WANG_INITIAL_LR,
            "learning_rate_schedule": "10^-4.23 / (8*x + 1)^1.5",
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "trunk_layers": self.trunk_layers,
            "head_layers": self.head_layers,
        }


class WangLearningRateSchedule:
    """Learning-rate annealing reported by Wang (2024).

    SB3 calls schedules with ``progress_remaining`` from 1 to 0, while Wang's
    training-progress variable x runs from 0 to 1.
    """

    def __call__(self, progress_remaining: float) -> float:
        x = 1.0 - float(progress_remaining)
        x = min(max(x, 0.0), 1.0)
        return float(
            WANG_INITIAL_LR / ((8.0 * x + 1.0) ** 1.5)
        )
