from __future__ import annotations

from collections import defaultdict
import json
import random
from typing import Any

import orjson

from poke_env.battle.battle import Battle
from poke_env.battle.pokemon import Pokemon
from poke_env.data import GenData
from poke_env.exceptions import ShowdownException
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
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



def pokemon_special_state(pokemon: Any) -> tuple[bool, str]:
    """Return Terastallization state in a version-tolerant way."""
    if pokemon is None:
        return False, ""
    is_tera = bool(
        getattr(pokemon, "is_terastallized", False)
        or getattr(pokemon, "terastallized", False)
    )
    tera_type = (
        getattr(pokemon, "tera_type", None)
        or getattr(pokemon, "terastallized_type", None)
    )
    return is_tera, enum_name(tera_type)


def order_special_flags(order: Any) -> tuple[bool, bool, bool, bool]:
    """Best-effort extraction of generation-specific order flags."""
    return (
        bool(getattr(order, "mega", False)),
        bool(getattr(order, "z_move", False) or getattr(order, "zmove", False)),
        bool(getattr(order, "dynamax", False)),
        bool(getattr(order, "terastallize", False)),
    )


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
    necessarily one call per turn, so these records are called *action requests*.

    Information policy:
    - ``battle.team`` contains the agent's own team.
    - ``battle.opponent_team`` is treated strictly as the currently revealed
      portion of the opposing team.
    - Hidden opponent slots are never reconstructed or exposed to an agent.
    """

    agent_name = "InstrumentedPlayer"

    def __init__(self, *args: Any, log_actions: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_actions = log_actions
        self.action_log: list[dict[str, Any]] = []
        self.action_request_counts: dict[str, int] = defaultdict(int)

    def _base_battle_format(self) -> str:
        return self._format.split("@@@", 1)[0]

    async def _handle_challenge_request(self, split_message):
        challenging_player = split_message[2].strip()

        if challenging_player == self.username:
            return

        if len(split_message) < 6:
            return

        received_format = split_message[5]
        base_format = self._base_battle_format()

        if received_format in (self._format, base_format):
            await self._challenge_queue.put(challenging_player)

    async def _update_challenges(self, split_message):
        challenges = orjson.loads(
            split_message[2]
        ).get("challengesFrom", {})

        base_format = self._base_battle_format()

        for user, format_ in challenges.items():
            if format_ in (self._format, base_format):
                await self._challenge_queue.put(user)

    async def _create_battle(self, split_message):
        """
        Workaround para poke-env #851.

        O Showdown usa apenas o formato base no nome da sala:
            gen9customgame

        enquanto o Player usa:
            gen9customgame@@@!teampreview

        Esta implementação é específica para os singles
        utilizados pelo PokéBench30.
        """

        base_format = self._base_battle_format()
        received_format = split_message[1]

        if received_format not in (
            base_format,
            f"{base_format}bo3",
        ):
            self.logger.critical(
                "Unmanaged battle initialisation message received: %s",
                split_message,
            )
            raise ShowdownException()

        battle_tag = "-".join(split_message)[1:]

        if battle_tag in self._battles:
            return self._battles[battle_tag]

        gen = GenData.from_format(base_format).gen

        battle = Battle(
            battle_tag=battle_tag,
            username=self.username,
            logger=self.logger,
            save_replays=self._save_replays,
            gen=gen,
        )

        # Registra o próprio time no objeto da batalha.
        if isinstance(self._team, ConstantTeambuilder):
            battle.teampreview_team = [
                Pokemon(
                    gen=gen,
                    teambuilder=tb_mon,
                )
                for tb_mon in self._team.team
            ]

        await self._battle_count_queue.put(None)

        async with self._battle_start_condition:
            self._battle_semaphore.release()
            self._battle_start_condition.notify_all()
            self._battles[battle_tag] = battle

        return battle

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


    def choose_random_core_order(self, battle: Any):
        """Choose uniformly among ordinary legal moves and switches.

        This intentionally excludes generation-specific gimmick activation
        (Mega/Z/Dynamax/Terastallization) so the scripted baselines are
        comparable until those mechanics are modeled explicitly.
        """
        moves = list(getattr(battle, "available_moves", []) or [])
        switches = list(getattr(battle, "available_switches", []) or [])

        candidates = [("move", move) for move in moves] + [
            ("switch", pokemon) for pokemon in switches
        ]

        if candidates:
            action_type, selected = random.choice(candidates)
            return self.create_order(selected), action_type

        # Extremely defensive fallback for unusual simulator requests.
        return self.choose_random_move(battle), "fallback"

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
        active_tera, active_tera_type = pokemon_special_state(active)
        opp_tera, opp_tera_type = pokemon_special_state(opponent)
        order_mega, order_z_move, order_dynamax, order_tera = order_special_flags(order)

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
            "active_terastallized": active_tera,
            "active_tera_type": active_tera_type,
            "active_boosts": self._json(getattr(active, "boosts", {}) if active else {}),
            "opponent_pokemon": getattr(opponent, "species", "") if opponent else "",
            "opponent_hp_fraction": hp_fraction(opponent),
            "opponent_status": enum_name(getattr(opponent, "status", None)) if opponent else "",
            "opponent_type_1": opp_t1,
            "opponent_type_2": opp_t2,
            "opponent_terastallized": opp_tera,
            "opponent_tera_type": opp_tera_type,
            "opponent_boosts": self._json(getattr(opponent, "boosts", {}) if opponent else {}),
            "own_team_size": own_known,
            "own_remaining": own_remaining,
            "own_fainted": own_fainted,
            "own_hp_sum": own_hp_sum,
            "opponent_revealed": opp_known,
            "opponent_revealed_remaining": opp_remaining,
            "opponent_revealed_fainted": opp_fainted,
            "opponent_revealed_hp_sum": opp_hp_sum,
            "action_type": action_type,
            "action": action,
            "voluntary_switch": action_type == "switch" and not forced,
            "forced_switch": action_type == "switch" and forced,
            "decision_reason": decision_reason,
            "order_mega": order_mega,
            "order_z_move": order_z_move,
            "order_dynamax": order_dynamax,
            "order_terastallize": order_tera,
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
