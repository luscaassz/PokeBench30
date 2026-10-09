from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


FEATURE_SCHEMA_VERSION = "ubr-1.0"

TYPE_NAMES = (
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE",
    "FIGHTING", "POISON", "GROUND", "FLYING", "PSYCHIC", "BUG",
    "ROCK", "GHOST", "DRAGON", "DARK", "STEEL", "FAIRY",
)

STATUS_NAMES = (
    "BRN", "FRZ", "PAR", "PSN", "SLP", "TOX",
)

BOOST_NAMES = (
    "atk", "def", "spa", "spd", "spe", "accuracy", "evasion",
)

MOVE_CATEGORY_NAMES = (
    "PHYSICAL", "SPECIAL", "STATUS",
)

MAX_TEAM_SIZE = 6
MAX_MOVES = 4

ACTION_LABELS = tuple(
    [f"move_slot_{i}" for i in range(1, MAX_MOVES + 1)]
    + [f"switch_team_slot_{i}" for i in range(1, MAX_TEAM_SIZE + 1)]
)


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    feature_names: tuple[str, ...]
    action_labels: tuple[str, ...]

    @property
    def observation_dim(self) -> int:
        return len(self.feature_names)

    @property
    def action_dim(self) -> int:
        return len(self.action_labels)

    @property
    def sha256(self) -> str:
        payload = {
            "version": self.version,
            "feature_names": self.feature_names,
            "action_labels": self.action_labels,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict:
        return {
            "schema_version": self.version,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "sha256": self.sha256,
            "feature_names": list(self.feature_names),
            "action_labels": list(self.action_labels),
        }


def _pokemon_slot_features(prefix: str) -> list[str]:
    names = [
        f"{prefix}.known",
        f"{prefix}.active",
        f"{prefix}.fainted",
        f"{prefix}.hp_fraction",
        f"{prefix}.level_norm",
    ]

    names.extend(
        f"{prefix}.type.{type_name}"
        for type_name in TYPE_NAMES
    )

    names.extend(
        f"{prefix}.status.{status_name}"
        for status_name in STATUS_NAMES
    )

    return names


def _move_slot_features(prefix: str) -> list[str]:
    names = [
        f"{prefix}.present",
        f"{prefix}.legal",
        f"{prefix}.base_power_norm",
        f"{prefix}.accuracy",
        f"{prefix}.priority_norm",
    ]

    names.extend(
        f"{prefix}.type.{type_name}"
        for type_name in TYPE_NAMES
    )

    names.extend(
        f"{prefix}.category.{category}"
        for category in MOVE_CATEGORY_NAMES
    )

    return names


def build_feature_schema() -> FeatureSchema:
    names: list[str] = [
        "global.generation_norm",
        "global.turn_norm",
        "global.force_switch",
        "global.own_remaining_fraction",
        "global.opponent_revealed_fraction",
        "global.opponent_revealed_remaining_fraction",
    ]

    names.extend(
        f"active_self.boost.{boost}"
        for boost in BOOST_NAMES
    )
    names.extend(
        f"active_opponent.boost.{boost}"
        for boost in BOOST_NAMES
    )

    for slot in range(1, MAX_TEAM_SIZE + 1):
        names.extend(
            _pokemon_slot_features(f"own_team.slot_{slot}")
        )

    for slot in range(1, MAX_TEAM_SIZE + 1):
        names.extend(
            _pokemon_slot_features(f"opponent_team.slot_{slot}")
        )

    for slot in range(1, MAX_MOVES + 1):
        names.extend(
            _move_slot_features(f"active_moves.slot_{slot}")
        )

    return FeatureSchema(
        version=FEATURE_SCHEMA_VERSION,
        feature_names=tuple(names),
        action_labels=ACTION_LABELS,
    )
