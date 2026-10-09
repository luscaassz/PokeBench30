from __future__ import annotations

from typing import Any

import numpy as np

from .schema import ACTION_LABELS, MAX_MOVES, MAX_TEAM_SIZE


ACTION_DIM = len(ACTION_LABELS)


class IllegalActionError(ValueError):
    pass


def _force_switch(battle: Any) -> bool:
    value = getattr(battle, "force_switch", False)
    if isinstance(value, (list, tuple)):
        return any(bool(item) for item in value)
    return bool(value)


def _move_id(move: Any) -> str:
    if move is None:
        return ""
    return str(
        getattr(move, "id", None)
        or getattr(move, "name", None)
        or move
    ).lower()


def _pokemon_ident(pokemon: Any) -> str:
    if pokemon is None:
        return ""

    for attr in ("ident", "name"):
        value = getattr(pokemon, attr, None)
        if value:
            return str(value).lower()

    species = getattr(pokemon, "species", None)
    if species:
        return str(species).lower()

    return str(id(pokemon))


def _same_pokemon(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if left is None or right is None:
        return False

    left_ident = _pokemon_ident(left)
    right_ident = _pokemon_ident(right)

    if left_ident and right_ident and left_ident == right_ident:
        return True

    left_species = str(getattr(left, "species", "")).lower()
    right_species = str(getattr(right, "species", "")).lower()

    if left_species and right_species and left_species == right_species:
        left_level = getattr(left, "level", None)
        right_level = getattr(right, "level", None)
        if left_level is None or right_level is None:
            return True
        return left_level == right_level

    return False


class UniversalActionSpace:
    """Generation-independent core action space.

    Indices 0-3:
        move slots 1-4

    Indices 4-9:
        own team slots 1-6

    The six team slots stay stable within one battle. The slot containing the
    active Pokémon is simply masked as illegal.

    Generation-specific gimmicks are intentionally outside v0.4's core action
    space because the scripted benchmark currently disables them.
    """

    action_labels = ACTION_LABELS
    action_dim = ACTION_DIM

    @staticmethod
    def own_team_slots(battle: Any) -> list[Any | None]:
        team = getattr(battle, "team", {}) or {}
        values = list(team.values()) if hasattr(team, "values") else list(team)

        active = getattr(battle, "active_pokemon", None)
        if active is not None and not any(
            _same_pokemon(active, pokemon)
            for pokemon in values
        ):
            values.insert(0, active)

        return (values[:MAX_TEAM_SIZE] + [None] * MAX_TEAM_SIZE)[:MAX_TEAM_SIZE]

    @staticmethod
    def opponent_team_slots(battle: Any) -> list[Any | None]:
        """Return only the opponent objects already revealed by poke-env.

        No attempt is made to reconstruct hidden team members.
        """
        team = getattr(battle, "opponent_team", {}) or {}
        values = list(team.values()) if hasattr(team, "values") else list(team)

        active = getattr(battle, "opponent_active_pokemon", None)
        if active is not None and not any(
            _same_pokemon(active, pokemon)
            for pokemon in values
        ):
            values.insert(0, active)

        return (values[:MAX_TEAM_SIZE] + [None] * MAX_TEAM_SIZE)[:MAX_TEAM_SIZE]

    @staticmethod
    def move_slots(battle: Any) -> list[Any | None]:
        active = getattr(battle, "active_pokemon", None)
        known_moves: list[Any] = []

        if active is not None:
            moves = getattr(active, "moves", {}) or {}
            known_moves = (
                list(moves.values())
                if hasattr(moves, "values")
                else list(moves)
            )

        available = list(getattr(battle, "available_moves", []) or [])

        # Add simulator-provided legal moves that are not currently present in
        # active.moves (for example special fallback moves).
        known_ids = {_move_id(move) for move in known_moves}
        for move in available:
            if _move_id(move) not in known_ids:
                known_moves.append(move)
                known_ids.add(_move_id(move))

        return (known_moves[:MAX_MOVES] + [None] * MAX_MOVES)[:MAX_MOVES]

    def legal_mask(self, battle: Any) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=np.int8)

        force_switch = _force_switch(battle)
        move_slots = self.move_slots(battle)
        available_moves = list(getattr(battle, "available_moves", []) or [])
        legal_move_ids = {_move_id(move) for move in available_moves}

        if not force_switch:
            for index, move in enumerate(move_slots):
                if move is not None and _move_id(move) in legal_move_ids:
                    mask[index] = 1

        team_slots = self.own_team_slots(battle)
        available_switches = list(
            getattr(battle, "available_switches", []) or []
        )

        for slot_index, pokemon in enumerate(team_slots):
            if pokemon is None:
                continue

            if any(
                _same_pokemon(pokemon, switch)
                for switch in available_switches
            ):
                mask[MAX_MOVES + slot_index] = 1

        return mask

    def object_for_action(self, battle: Any, action_index: int) -> Any:
        if not 0 <= int(action_index) < self.action_dim:
            raise IllegalActionError(
                f"Action {action_index} is outside [0, {self.action_dim - 1}]."
            )

        mask = self.legal_mask(battle)
        if mask[int(action_index)] != 1:
            raise IllegalActionError(
                f"Action {action_index} ({self.action_labels[int(action_index)]}) "
                "is not legal in the current state."
            )

        if action_index < MAX_MOVES:
            selected = self.move_slots(battle)[action_index]
        else:
            selected = self.own_team_slots(battle)[
                action_index - MAX_MOVES
            ]

        if selected is None:
            raise IllegalActionError(
                f"Legal-mask inconsistency for action {action_index}."
            )

        return selected

    def index_for_selected(self, battle: Any, selected: Any) -> int | None:
        if selected is None:
            return None

        selected_move_id = _move_id(selected)
        if getattr(selected, "base_power", None) is not None or hasattr(
            selected, "category"
        ):
            for index, move in enumerate(self.move_slots(battle)):
                if move is selected:
                    return index
                if move is not None and _move_id(move) == selected_move_id:
                    return index

        if getattr(selected, "species", None) is not None:
            for slot_index, pokemon in enumerate(self.own_team_slots(battle)):
                if _same_pokemon(pokemon, selected):
                    return MAX_MOVES + slot_index

        return None

    def index_for_order(self, battle: Any, order: Any) -> int | None:
        return self.index_for_selected(
            battle,
            getattr(order, "order", None),
        )

    def label(self, action_index: int | None) -> str:
        if action_index is None:
            return ""
        if 0 <= int(action_index) < self.action_dim:
            return self.action_labels[int(action_index)]
        return ""
