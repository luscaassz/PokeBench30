from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .action_space import UniversalActionSpace, _same_pokemon
from .schema import (
    BOOST_NAMES,
    MAX_MOVES,
    MAX_TEAM_SIZE,
    MOVE_CATEGORY_NAMES,
    STATUS_NAMES,
    TYPE_NAMES,
    FeatureSchema,
    build_feature_schema,
)


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read a poke-env property that may be absent from transient commands.

    Gen I pseudo-moves such as ``recharge`` / ``fight`` can be represented by
    Move objects whose data entry omits ordinary keys such as ``priority``.
    Some poke-env properties raise KeyError instead of AttributeError in that
    situation, so plain getattr(..., default) is not sufficient.
    """
    try:
        return getattr(obj, name)
    except (AttributeError, KeyError, TypeError, ValueError):
        return default


def _enum_name(value: Any) -> str:
    if value is None:
        return ""
    name = getattr(value, "name", None)
    return str(name if name is not None else value).upper()


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
        not bool(getattr(pokemon, "fainted", False))
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
    boosts = getattr(pokemon, "boosts", {}) or {}
    try:
        value = float(boosts.get(key, 0.0))
    except (AttributeError, TypeError, ValueError):
        value = 0.0
    return _clip(value / 6.0, -1.0, 1.0)


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
    """Core universal battle representation for Gen I / IV / IX.

    Design goals:
    - fixed dimensionality across generations;
    - species-agnostic features to support cross-generation transfer;
    - no hidden-opponent reconstruction;
    - normalized numeric features;
    - legal-action mask paired with every observation.

    Deliberately omitted from UBR 1.1:
    weather, terrain, hazards, volatile effects, abilities/items and
    generation-specific gimmicks. Those can be layered onto future schema
    versions without silently changing UBR 1.1.
    """

    def __init__(
        self,
        action_space: UniversalActionSpace | None = None,
        schema: FeatureSchema | None = None,
    ) -> None:
        self.action_space = action_space or UniversalActionSpace()
        self.schema = schema or build_feature_schema()

    @staticmethod
    def _generation(battle: Any) -> int:
        for attr in ("gen", "generation"):
            value = getattr(battle, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass

        battle_format = str(getattr(battle, "format", "") or "")
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
            1.0 if bool(getattr(pokemon, "fainted", False)) else 0.0,
            _hp_fraction(pokemon),
        ]

        try:
            level = float(getattr(pokemon, "level", 100) or 100)
        except (TypeError, ValueError):
            level = 100.0
        values.append(_clip(level / 100.0, 0.0, 1.0))

        pokemon_types = {
            _enum_name(getattr(pokemon, "type_1", None)),
            _enum_name(getattr(pokemon, "type_2", None)),
        }

        values.extend(
            1.0 if type_name in pokemon_types else 0.0
            for type_name in TYPE_NAMES
        )

        status = _enum_name(getattr(pokemon, "status", None))
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
        force_switch = getattr(battle, "force_switch", False)
        if isinstance(force_switch, (list, tuple)):
            force_switch = any(bool(value) for value in force_switch)
        else:
            force_switch = bool(force_switch)

        own_team = getattr(battle, "team", {}) or {}
        opponent_team = getattr(battle, "opponent_team", {}) or {}

        own_slots = self.action_space.own_team_slots(battle)
        opponent_slots = self.action_space.opponent_team_slots(battle)
        move_slots = self.action_space.move_slots(battle)
        legal_mask = self.action_space.legal_mask(battle)

        try:
            turn = float(getattr(battle, "turn", 0.0) or 0.0)
        except (TypeError, ValueError):
            turn = 0.0

        vector: list[float] = [
            _clip(generation / 9.0, 0.0, 1.0),
            _clip(turn / 100.0, 0.0, 1.0),
            1.0 if force_switch else 0.0,
            _remaining_fraction(own_team),
            _known_fraction(opponent_team),
            _remaining_fraction(opponent_team),
        ]

        active = getattr(battle, "active_pokemon", None)
        opponent_active = getattr(
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

        array = np.asarray(vector, dtype=np.float32)

        if array.shape != (self.schema.observation_dim,):
            raise RuntimeError(
                "Universal representation dimension drift: "
                f"expected {self.schema.observation_dim}, "
                f"got {array.shape}."
            )

        metadata = {
            "generation": generation,
            "turn": int(turn),
            "battle_tag": str(
                getattr(battle, "battle_tag", "")
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
        }

        return EncodedObservation(
            vector=array,
            legal_action_mask=legal_mask,
            schema_sha256=self.schema.sha256,
            metadata=metadata,
        )
