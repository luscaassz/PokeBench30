from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    """Dense reward weights used by the first PokéBench30 RL baseline.

    The reward is computed as a state-value difference by poke-env's
    ``reward_computing_helper``. Positive changes for the learner and negative
    changes for the opponent are symmetric.
    """

    fainted_value: float = 2.0
    hp_value: float = 1.0
    status_value: float = 0.5
    victory_value: float = 30.0

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "RewardConfig":
        mapping = mapping or {}
        return cls(
            fainted_value=float(mapping.get("fainted_value", 2.0)),
            hp_value=float(mapping.get("hp_value", 1.0)),
            status_value=float(mapping.get("status_value", 0.5)),
            victory_value=float(mapping.get("victory_value", 30.0)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "fainted_value": self.fainted_value,
            "hp_value": self.hp_value,
            "status_value": self.status_value,
            "victory_value": self.victory_value,
        }
