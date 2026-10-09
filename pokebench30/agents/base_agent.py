from __future__ import annotations

from collections import defaultdict
import json
from typing import Any

from poke_env.player import Player


def enum_name(value: Any) -> str:
    if value is None:
        return ""
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def pokemon_types(pokemon: Any) -> tuple[str, str]:
    if pokemon is None:
        return "", ""
    return (
        enum_name(getattr(pokemon, "type_1", None)),
        enum_name(getattr(pokemon, "type_2", None)),
    )


def hp_fraction(pokemon: Any) -> float | None:
    if pokemon is None:
        return None
    try:
        return round(float(pokemon.current_hp_fraction), 6)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return None


def is_force_switch(battle: Any) -> bool:
    value = getattr(battle, "force_switch", False)
    if isinstance(value, (list, tuple)):
        return any(bool(item) for item in value)
    return bool(value)


def team_snapshot(team: Any) -> tuple[int, int, int, float]:
    """Return known, remaining, fainted and summed HP fraction."""
    if not team:
        return 0, 0, 0, 0.0
    values = list(team.values()) if hasattr(team, "values") else list(team)
    known = len(values)
    fainted = sum(bool(getattr(mon, "fainted", False)) for mon in values)
    remaining = known - fainted
    hp_sum = sum(float(getattr(mon, "current_hp_fraction", 0.0) or 0.0) for mon in values)
    return known, remaining, fainted, round(hp_sum, 6)


class InstrumentedPlayer(Player):
    """Base player with state-rich action-request logging.

    ``choose_move`` is called when Showdown requests an action. This is not
    necessarily one call per turn, so v0.2 explicitly calls these records
    *action requests* rather than turns/decisions.
    """

    agent_name = "InstrumentedPlayer"

    def __init__(self, *args: Any, log_actions: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_actions = log_actions
        self.action_log: list[dict[str, Any]] = []
        self.action_request_counts: dict[str, int] = defaultdict(int)

    @staticmethod
    def _action_info(order: Any) -> tuple[str, str, Any]:
        selected = getattr(order, "order", None)
        if selected is None:
            return "other", order.__class__.__name__, None

        move_id = getattr(selected, "id", None)
        if move_id is not None:
            return "move", str(move_id), selected

        species = getattr(selected, "species", None)
        if species is not None:
            return "switch", str(species), selected

        return "other", str(selected), selected

    @staticmethod
    def _json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return json.dumps(str(value), ensure_ascii=False)

    def record_order(
        self,
        battle: Any,
        order: Any,
        *,
        heuristic_score: float | None = None,
        decision_reason: str = "",
        current_matchup_score: float | None = None,
        target_matchup_score: float | None = None,
    ) -> Any:
        battle_tag = getattr(battle, "battle_tag", "unknown")
        self.action_request_counts[battle_tag] += 1
        request_index = self.action_request_counts[battle_tag]

        if not self.log_actions:
            return order

        active = getattr(battle, "active_pokemon", None)
        opponent = getattr(battle, "opponent_active_pokemon", None)
        own_known, own_remaining, own_fainted, own_hp_sum = team_snapshot(
            getattr(battle, "team", {})
        )
        opp_known, opp_remaining, opp_fainted, opp_hp_sum = team_snapshot(
            getattr(battle, "opponent_team", {})
        )

        action_type, action, selected = self._action_info(order)
        forced = is_force_switch(battle)
        active_t1, active_t2 = pokemon_types(active)
        opp_t1, opp_t2 = pokemon_types(opponent)

        row: dict[str, Any] = {
            "battle_tag": battle_tag,
            "request_index": request_index,
            "turn": getattr(battle, "turn", None),
            "agent": self.agent_name,
            "forced_switch_request": forced,
            "available_moves": len(getattr(battle, "available_moves", []) or []),
            "available_switches": len(getattr(battle, "available_switches", []) or []),
            "active_pokemon": getattr(active, "species", "") if active else "",
            "active_hp_fraction": hp_fraction(active),
            "active_status": enum_name(getattr(active, "status", None)) if active else "",
            "active_type_1": active_t1,
            "active_type_2": active_t2,
            "active_boosts": self._json(getattr(active, "boosts", {}) if active else {}),
            "opponent_pokemon": getattr(opponent, "species", "") if opponent else "",
            "opponent_hp_fraction": hp_fraction(opponent),
            "opponent_status": enum_name(getattr(opponent, "status", None)) if opponent else "",
            "opponent_type_1": opp_t1,
            "opponent_type_2": opp_t2,
            "opponent_boosts": self._json(getattr(opponent, "boosts", {}) if opponent else {}),
            "own_team_known": own_known,
            "own_remaining_known": own_remaining,
            "own_fainted_known": own_fainted,
            "own_hp_sum_known": own_hp_sum,
            "opponent_team_known": opp_known,
            "opponent_remaining_known": opp_remaining,
            "opponent_fainted_known": opp_fainted,
            "opponent_hp_sum_known": opp_hp_sum,
            "action_type": action_type,
            "action": action,
            "voluntary_switch": action_type == "switch" and not forced,
            "forced_switch": action_type == "switch" and forced,
            "decision_reason": decision_reason,
            "heuristic_score": None if heuristic_score is None else round(float(heuristic_score), 6),
            "current_matchup_score": None if current_matchup_score is None else round(float(current_matchup_score), 6),
            "target_matchup_score": None if target_matchup_score is None else round(float(target_matchup_score), 6),
            "move_base_power": None,
            "move_accuracy": None,
            "move_type": "",
            "move_category": "",
            "move_stab": None,
            "move_type_effectiveness": None,
            "switch_target_hp_fraction": None,
            "switch_target_type_1": "",
            "switch_target_type_2": "",
        }

        if action_type == "move" and selected is not None:
            row["move_base_power"] = getattr(selected, "base_power", None)
            row["move_accuracy"] = getattr(selected, "accuracy", None)
            row["move_type"] = enum_name(getattr(selected, "type", None))
            row["move_category"] = enum_name(getattr(selected, "category", None))

            if active is not None:
                active_types = {
                    value
                    for value in (
                        getattr(active, "type_1", None),
                        getattr(active, "type_2", None),
                    )
                    if value is not None
                }
                row["move_stab"] = 1.5 if getattr(selected, "type", None) in active_types else 1.0

            if opponent is not None:
                try:
                    row["move_type_effectiveness"] = float(opponent.damage_multiplier(selected))
                except (AttributeError, TypeError, ValueError):
                    pass

        elif action_type == "switch" and selected is not None:
            row["switch_target_hp_fraction"] = hp_fraction(selected)
            s1, s2 = pokemon_types(selected)
            row["switch_target_type_1"] = s1
            row["switch_target_type_2"] = s2

        self.action_log.append(row)
        return order
