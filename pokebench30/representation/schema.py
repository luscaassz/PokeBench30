from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


FEATURE_SCHEMA_VERSION = "ubr-1.2"

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

# UBR 1.2 appends a compact battle-context block to the 472-dimensional
# UBR 1.1 core. The prefix remains unchanged for easier ablations.
SIDE_CONDITION_FEATURES = (
    "stealth_rock",
    "spikes",
    "toxic_spikes",
    "sticky_web",
    "reflect",
    "light_screen",
    "aurora_veil",
    "tailwind",
)

WEATHER_FEATURES = (
    "rain",
    "sun",
    "sandstorm",
    "hail",
    "snow",
    "primordial_sea",
    "desolate_land",
    "delta_stream",
)

FIELD_FEATURES = (
    "electric_terrain",
    "grassy_terrain",
    "misty_terrain",
    "psychic_terrain",
    "trick_room",
    "gravity",
)

VOLATILE_EFFECT_FEATURES = (
    "taunt",
    "encore",
    "substitute",
    "confusion",
    "leech_seed",
    "disable",
    "yawn",
    "protect",
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
    # ---- UBR 1.1-compatible 472-dimensional prefix ----
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

    # ---- UBR 1.2 battle-context extension: 46 features ----
    for side in ("own_side", "opponent_side"):
        names.extend(
            f"battle_context.{side}.{feature}"
            for feature in SIDE_CONDITION_FEATURES
        )

    names.extend(
        f"battle_context.weather.{feature}"
        for feature in WEATHER_FEATURES
    )

    names.extend(
        f"battle_context.field.{feature}"
        for feature in FIELD_FEATURES
    )

    for perspective in ("active_self", "active_opponent"):
        names.extend(
            f"battle_context.{perspective}.effect.{feature}"
            for feature in VOLATILE_EFFECT_FEATURES
        )

    return FeatureSchema(
        version=FEATURE_SCHEMA_VERSION,
        feature_names=tuple(names),
        action_labels=ACTION_LABELS,
    )
