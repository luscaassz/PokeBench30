from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .action_space import UniversalActionSpace, _same_pokemon
from .schema import (
    BOOST_NAMES,
    FIELD_FEATURES,
    MAX_MOVES,
    MAX_TEAM_SIZE,
    MOVE_CATEGORY_NAMES,
    SIDE_CONDITION_FEATURES,
    STATUS_NAMES,
    TYPE_NAMES,
    VOLATILE_EFFECT_FEATURES,
    WEATHER_FEATURES,
    FeatureSchema,
    build_feature_schema,
)


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read a property that may be absent or backed by incomplete data."""
    try:
        return getattr(obj, name)
    except (AttributeError, KeyError, TypeError, ValueError):
        return default


def _enum_name(value: Any) -> str:
    if value is None:
        return ""
    name = _safe_attr(value, "name", None)
    return str(name if name is not None else value).upper()


def _enum_token(value: Any) -> str:
    """Normalize enum/string names across poke-env / Showdown naming styles."""
    return "".join(
        char
        for char in _enum_name(value)
        if char.isalnum()
    )


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(float(value), low), high))


def _hp_fraction(pokemon: Any) -> float:
    if pokemon is None:
        return 0.0
    try:
        return _clip(float(pokemon.current_hp_fraction), 0.0, 1.0)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _accuracy_fraction(value: Any) -> float:
    if value is None or value is True:
        return 1.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0

    if value <= 0:
        return 1.0
    if value > 1:
        value /= 100.0
    return _clip(value, 0.0, 1.0)


def _team_values(team: Any) -> list[Any]:
    if not team:
        return []
    return list(team.values()) if hasattr(team, "values") else list(team)


def _remaining_fraction(team: Any) -> float:
    values = _team_values(team)
    if not values:
        return 0.0
    remaining = sum(
        not bool(_safe_attr(pokemon, "fainted", False))
        for pokemon in values
    )
    return _clip(remaining / MAX_TEAM_SIZE, 0.0, 1.0)


def _known_fraction(team: Any) -> float:
    return _clip(
        len(_team_values(team)) / MAX_TEAM_SIZE,
        0.0,
        1.0,
    )


def _boost_value(pokemon: Any, key: str) -> float:
    if pokemon is None:
        return 0.0
    boosts = _safe_attr(pokemon, "boosts", {}) or {}
    try:
        value = float(boosts.get(key, 0.0))
    except (AttributeError, TypeError, ValueError):
        value = 0.0
    return _clip(value / 6.0, -1.0, 1.0)


def _mapping_items(mapping: Any) -> list[tuple[Any, Any]]:
    if not mapping:
        return []
    if hasattr(mapping, "items"):
        try:
            return list(mapping.items())
        except (AttributeError, TypeError, ValueError):
            return []
    try:
        return [(value, 1) for value in mapping]
    except TypeError:
        return []


def _mapping_tokens(mapping: Any) -> set[str]:
    return {
        _enum_token(key)
        for key, _ in _mapping_items(mapping)
        if _enum_token(key)
    }


def _mapping_value_for_tokens(
    mapping: Any,
    accepted_tokens: set[str],
) -> Any | None:
    for key, value in _mapping_items(mapping):
        if _enum_token(key) in accepted_tokens:
            return value
    return None


SIDE_CONDITION_ALIASES: dict[str, set[str]] = {
    "stealth_rock": {"STEALTHROCK"},
    "spikes": {"SPIKES"},
    "toxic_spikes": {"TOXICSPIKES"},
    "sticky_web": {"STICKYWEB"},
    "reflect": {"REFLECT"},
    "light_screen": {"LIGHTSCREEN"},
    "aurora_veil": {"AURORAVEIL"},
    "tailwind": {"TAILWIND"},
}

SIDE_CONDITION_MAX_LAYERS = {
    "spikes": 3,
    "toxic_spikes": 2,
}

WEATHER_ALIASES: dict[str, set[str]] = {
    "rain": {"RAINDANCE", "RAIN"},
    "sun": {"SUNNYDAY", "SUN"},
    "sandstorm": {"SANDSTORM"},
    "hail": {"HAIL"},
    "snow": {"SNOW"},
    "primordial_sea": {"PRIMORDIALSEA"},
    "desolate_land": {"DESOLATELAND"},
    "delta_stream": {"DELTASTREAM"},
}

FIELD_ALIASES: dict[str, set[str]] = {
    "electric_terrain": {"ELECTRICTERRAIN"},
    "grassy_terrain": {"GRASSYTERRAIN"},
    "misty_terrain": {"MISTYTERRAIN"},
    "psychic_terrain": {"PSYCHICTERRAIN"},
    "trick_room": {"TRICKROOM"},
    "gravity": {"GRAVITY"},
}

VOLATILE_EFFECT_ALIASES: dict[str, set[str]] = {
    "taunt": {"TAUNT"},
    "encore": {"ENCORE"},
    "substitute": {"SUBSTITUTE"},
    "confusion": {"CONFUSION"},
    "leech_seed": {"LEECHSEED"},
    "disable": {"DISABLE"},
    "yawn": {"YAWN"},
    "protect": {"PROTECT"},
}


def _side_condition_features(mapping: Any) -> list[float]:
    """Encode selected side conditions.

    Stackable hazards use normalized layer counts. Other tracked conditions
    use binary presence. poke-env documents stackable side-condition values as
    layer counts and non-stackable values as setup-turn identifiers.
    """
    values: list[float] = []

    for feature in SIDE_CONDITION_FEATURES:
        aliases = SIDE_CONDITION_ALIASES[feature]
        raw = _mapping_value_for_tokens(mapping, aliases)

        if raw is None:
            values.append(0.0)
            continue

        max_layers = SIDE_CONDITION_MAX_LAYERS.get(feature)
        if max_layers is None:
            values.append(1.0)
            continue

        try:
            layers = float(raw)
        except (TypeError, ValueError):
            layers = 1.0

        values.append(
            _clip(layers / float(max_layers), 0.0, 1.0)
        )

    return values


def _categorical_context_features(
    mapping: Any,
    names: tuple[str, ...],
    aliases: dict[str, set[str]],
) -> list[float]:
    tokens = _mapping_tokens(mapping)
    return [
        1.0 if bool(tokens.intersection(aliases[name])) else 0.0
        for name in names
    ]


def _volatile_features(pokemon: Any) -> list[float]:
    effects = _safe_attr(pokemon, "effects", {}) if pokemon is not None else {}
    return _categorical_context_features(
        effects,
        VOLATILE_EFFECT_FEATURES,
        VOLATILE_EFFECT_ALIASES,
    )


@dataclass(frozen=True)
class EncodedObservation:
    vector: np.ndarray
    legal_action_mask: np.ndarray
    schema_sha256: str
    metadata: dict[str, Any]

    def to_json_record(self) -> dict[str, Any]:
        return {
            "schema_sha256": self.schema_sha256,
            "metadata": self.metadata,
            "vector": [float(value) for value in self.vector],
            "legal_action_mask": [
                int(value) for value in self.legal_action_mask
            ],
        }


class UniversalStateEncoder:
    """Universal battle representation for Gen I / IV / IX.

    UBR 1.2 preserves the full UBR 1.1 core prefix and appends public
    battle-context information that is important for learned policies:
    selected side conditions, weather, terrain/field effects, and selected
    volatile effects on the currently active Pokémon.

    Design goals:
    - fixed dimensionality across generations;
    - species-agnostic features to support later transfer experiments;
    - no hidden-opponent reconstruction;
    - normalized numeric features in [-1, 1];
    - legal-action mask paired with every observation;
    - only agent-visible public battle state.

    Still deliberately omitted from UBR 1.2 include species identity,
    items/abilities, PP, known opponent move history, many rare side/effect
    states, and generation-specific gimmick actions.
    """

    def __init__(
        self,
        action_space: UniversalActionSpace | None = None,
        schema: FeatureSchema | None = None,
        representation_version: str | None = None,
    ) -> None:
        if schema is not None and representation_version is not None:
            if schema.version != representation_version:
                raise ValueError(
                    "schema.version and representation_version disagree: "
                    f"{schema.version!r} != {representation_version!r}."
                )

        self.action_space = action_space or UniversalActionSpace()
        self.schema = schema or build_feature_schema(
            representation_version or "ubr-1.2"
        )

    @staticmethod
    def _generation(battle: Any) -> int:
        for attr in ("gen", "generation"):
            value = _safe_attr(battle, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass

        battle_format = str(_safe_attr(battle, "format", "") or "")
        if battle_format.startswith("gen"):
            digits = ""
            for char in battle_format[3:]:
                if char.isdigit():
                    digits += char
                else:
                    break
            if digits:
                return int(digits)

        return 0

    @staticmethod
    def _pokemon_features(
        pokemon: Any,
        active: Any,
    ) -> list[float]:
        if pokemon is None:
            # Unknown / unused slots are exactly zero-filled.
            return [
                0.0
                for _ in range(
                    5 + len(TYPE_NAMES) + len(STATUS_NAMES)
                )
            ]

        values: list[float] = [
            1.0,  # known
            1.0 if _same_pokemon(pokemon, active) else 0.0,
            1.0 if bool(_safe_attr(pokemon, "fainted", False)) else 0.0,
            _hp_fraction(pokemon),
        ]

        try:
            level = float(_safe_attr(pokemon, "level", 100) or 100)
        except (TypeError, ValueError):
            level = 100.0
        values.append(_clip(level / 100.0, 0.0, 1.0))

        pokemon_types = {
            _enum_name(_safe_attr(pokemon, "type_1", None)),
            _enum_name(_safe_attr(pokemon, "type_2", None)),
        }

        values.extend(
            1.0 if type_name in pokemon_types else 0.0
            for type_name in TYPE_NAMES
        )

        status = _enum_name(_safe_attr(pokemon, "status", None))
        values.extend(
            1.0 if status == status_name else 0.0
            for status_name in STATUS_NAMES
        )

        return values

    @staticmethod
    def _move_features(
        move: Any,
        legal: bool,
    ) -> list[float]:
        if move is None:
            return [
                0.0
                for _ in range(
                    5 + len(TYPE_NAMES) + len(MOVE_CATEGORY_NAMES)
                )
            ]

        try:
            base_power = float(_safe_attr(move, "base_power", 0.0) or 0.0)
        except (TypeError, ValueError):
            base_power = 0.0

        try:
            priority = float(_safe_attr(move, "priority", 0.0) or 0.0)
        except (TypeError, ValueError):
            priority = 0.0

        values: list[float] = [
            1.0,
            1.0 if legal else 0.0,
            _clip(base_power / 250.0, 0.0, 1.0),
            _accuracy_fraction(_safe_attr(move, "accuracy", 1.0)),
            _clip(priority / 7.0, -1.0, 1.0),
        ]

        move_type = _enum_name(_safe_attr(move, "type", None))
        values.extend(
            1.0 if move_type == type_name else 0.0
            for type_name in TYPE_NAMES
        )

        category = _enum_name(_safe_attr(move, "category", None))
        values.extend(
            1.0 if category == name else 0.0
            for name in MOVE_CATEGORY_NAMES
        )

        return values

    def encode(self, battle: Any) -> EncodedObservation:
        generation = self._generation(battle)
        force_switch = _safe_attr(battle, "force_switch", False)
        if isinstance(force_switch, (list, tuple)):
            force_switch = any(bool(value) for value in force_switch)
        else:
            force_switch = bool(force_switch)

        own_team = _safe_attr(battle, "team", {}) or {}
        opponent_team = _safe_attr(battle, "opponent_team", {}) or {}

        own_slots = self.action_space.own_team_slots(battle)
        opponent_slots = self.action_space.opponent_team_slots(battle)
        move_slots = self.action_space.move_slots(battle)
        legal_mask = self.action_space.legal_mask(battle)

        try:
            turn = float(_safe_attr(battle, "turn", 0.0) or 0.0)
        except (TypeError, ValueError):
            turn = 0.0

        # ---- UBR 1.1-compatible prefix ----
        vector: list[float] = [
            _clip(generation / 9.0, 0.0, 1.0),
            _clip(turn / 100.0, 0.0, 1.0),
            1.0 if force_switch else 0.0,
            _remaining_fraction(own_team),
            _known_fraction(opponent_team),
            _remaining_fraction(opponent_team),
        ]

        active = _safe_attr(battle, "active_pokemon", None)
        opponent_active = _safe_attr(
            battle,
            "opponent_active_pokemon",
            None,
        )

        vector.extend(
            _boost_value(active, key)
            for key in BOOST_NAMES
        )
        vector.extend(
            _boost_value(opponent_active, key)
            for key in BOOST_NAMES
        )

        for pokemon in own_slots:
            vector.extend(
                self._pokemon_features(pokemon, active)
            )

        for pokemon in opponent_slots:
            vector.extend(
                self._pokemon_features(
                    pokemon,
                    opponent_active,
                )
            )

        for move_index, move in enumerate(move_slots):
            vector.extend(
                self._move_features(
                    move,
                    bool(legal_mask[move_index]),
                )
            )

        # ---- Optional UBR 1.2 battle-context extension ----
        own_side_conditions = _safe_attr(
            battle,
            "side_conditions",
            {},
        ) or {}
        opponent_side_conditions = _safe_attr(
            battle,
            "opponent_side_conditions",
            {},
        ) or {}
        weather = _safe_attr(battle, "weather", {}) or {}
        fields = _safe_attr(battle, "fields", {}) or {}

        if self.schema.version == "ubr-1.2":
            vector.extend(
                _side_condition_features(own_side_conditions)
            )
            vector.extend(
                _side_condition_features(opponent_side_conditions)
            )
            vector.extend(
                _categorical_context_features(
                    weather,
                    WEATHER_FEATURES,
                    WEATHER_ALIASES,
                )
            )
            vector.extend(
                _categorical_context_features(
                    fields,
                    FIELD_FEATURES,
                    FIELD_ALIASES,
                )
            )
            vector.extend(_volatile_features(active))
            vector.extend(_volatile_features(opponent_active))

        array = np.asarray(vector, dtype=np.float32)

        if array.shape != (self.schema.observation_dim,):
            raise RuntimeError(
                "Universal representation dimension drift: "
                f"expected {self.schema.observation_dim}, "
                f"got {array.shape}."
            )

        metadata = {
            "representation_version": self.schema.version,
            "generation": generation,
            "turn": int(turn),
            "battle_tag": str(
                _safe_attr(battle, "battle_tag", "")
                or ""
            ),
            "own_team_known": len(
                [slot for slot in own_slots if slot is not None]
            ),
            "opponent_revealed": len(
                [
                    slot
                    for slot in opponent_slots
                    if slot is not None
                ]
            ),
            "force_switch": force_switch,
            "battle_context": {
                "own_side_conditions_tracked": int(
                    sum(value > 0.0 for value in _side_condition_features(own_side_conditions))
                ),
                "opponent_side_conditions_tracked": int(
                    sum(value > 0.0 for value in _side_condition_features(opponent_side_conditions))
                ),
                "weather_tracked": int(
                    sum(
                        _categorical_context_features(
                            weather,
                            WEATHER_FEATURES,
                            WEATHER_ALIASES,
                        )
                    )
                ),
                "fields_tracked": int(
                    sum(
                        _categorical_context_features(
                            fields,
                            FIELD_FEATURES,
                            FIELD_ALIASES,
                        )
                    )
                ),
                "active_self_effects_tracked": int(
                    sum(_volatile_features(active))
                ),
                "active_opponent_effects_tracked": int(
                    sum(_volatile_features(opponent_active))
                ),
            },
        }

        return EncodedObservation(
            vector=array,
            legal_action_mask=legal_mask,
            schema_sha256=self.schema.sha256,
            metadata=metadata,
        )
