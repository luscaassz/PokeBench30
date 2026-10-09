from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class PolicySpec:
    name: str
    family: str
    history_len: int
    net_arch: tuple[int, ...]
    temporal_hidden_size: int | None = None
    temporal_layers: int = 1
    description: str = ""

    def model_input_dim(self, ubr_dim: int) -> int:
        return int(ubr_dim) * int(self.history_len)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


POLICY_SPECS: dict[str, PolicySpec] = {
    "mlp_baseline": PolicySpec(
        name="mlp_baseline",
        family="mlp",
        history_len=1,
        net_arch=(256, 128),
        description="v0.5-v0.5.4 MaskablePPO baseline MLP.",
    ),
    "mlp_large": PolicySpec(
        name="mlp_large",
        family="mlp",
        history_len=1,
        net_arch=(512, 256, 128),
        description="Larger feed-forward MLP capacity control.",
    ),
    "history_mlp": PolicySpec(
        name="history_mlp",
        family="history_mlp",
        history_len=4,
        net_arch=(512, 256, 128),
        description=(
            "Four-decision UBR window flattened into an MLP. "
            "Controls for temporal information without an LSTM inductive bias."
        ),
    ),
    "history_lstm": PolicySpec(
        name="history_lstm",
        family="history_lstm",
        history_len=4,
        net_arch=(128,),
        temporal_hidden_size=256,
        temporal_layers=1,
        description=(
            "Four-decision UBR window encoded by an LSTM, then actor/critic heads. "
            "Uses MaskablePPO and therefore preserves strict invalid-action masking."
        ),
    ),
}

DEFAULT_POLICY_ARCHITECTURES = [
    "mlp_baseline",
    "mlp_large",
    "history_mlp",
    "history_lstm",
]


def get_policy_spec(name: str) -> PolicySpec:
    try:
        return POLICY_SPECS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown policy architecture {name!r}. "
            f"Available: {sorted(POLICY_SPECS)}"
        ) from exc


def validate_architectures(names: Iterable[str]) -> list[str]:
    values = [str(name) for name in names]
    if not values:
        raise ValueError("At least one policy architecture is required.")
    if len(set(values)) != len(values):
        raise ValueError("Policy architectures must be unique.")
    for name in values:
        get_policy_spec(name)
    return values


def padded_history(
    vectors: Iterable[np.ndarray],
    *,
    history_len: int,
    base_dim: int,
) -> np.ndarray:
    """Return oldest->newest flattened history with zero left-padding."""
    if history_len <= 0:
        raise ValueError("history_len must be > 0")
    if base_dim <= 0:
        raise ValueError("base_dim must be > 0")

    cleaned: list[np.ndarray] = []
    for vector in vectors:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.shape != (base_dim,):
            raise ValueError(
                f"Expected history vector shape {(base_dim,)}, got {arr.shape}."
            )
        cleaned.append(arr)

    cleaned = cleaned[-history_len:]
    missing = history_len - len(cleaned)
    padded = [np.zeros(base_dim, dtype=np.float32) for _ in range(missing)]
    padded.extend(cleaned)
    return np.concatenate(padded, axis=0).astype(np.float32, copy=False)
