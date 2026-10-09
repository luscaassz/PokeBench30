from __future__ import annotations

import random

from poke_env.battle import AbstractBattle

from .base_agent import hp_fraction, is_force_switch
from .type_aware_agent import TypeAwareAgent


class SwitchAwareAgent(TypeAwareAgent):
    """Type-aware attacker with transparent strategic voluntary switching."""

    agent_name = "SwitchAwareAgent"
    SWITCH_IMPROVEMENT_THRESHOLD = 1.0
    LOW_HP_THRESHOLD = 0.20
    SWITCH_COOLDOWN_TURNS = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_switch_from = {}
        self._last_switch_turn = {}

    @staticmethod
    def _types(pokemon):
        if pokemon is None:
            return []
        return [
            value
            for value in (
                getattr(pokemon, "type_1", None),
                getattr(pokemon, "type_2", None),
            )
            if value is not None
        ]

    @classmethod
    def matchup_score(cls, candidate, opponent) -> float:
        if candidate is None or opponent is None:
            return 0.0

        candidate_types = cls._types(candidate)
        opponent_types = cls._types(opponent)

        offensive = 1.0
        if candidate_types:
            values = []
            for type_ in candidate_types:
                try:
                    values.append(float(opponent.damage_multiplier(type_)))
                except (AttributeError, TypeError, ValueError):
                    pass
            if values:
                offensive = max(values)

        defensive_risk = 1.0
        if opponent_types:
            values = []
            for type_ in opponent_types:
                try:
                    values.append(float(candidate.damage_multiplier(type_)))
                except (AttributeError, TypeError, ValueError):
                    pass
            if values:
                defensive_risk = max(values)

        hp_bonus = 0.25 * float(hp_fraction(candidate) or 0.0)
        return float(offensive - defensive_risk + hp_bonus)

    @staticmethod
    def _species(pokemon):
        if pokemon is None:
            return None
        return str(getattr(pokemon, "species", ""))

    def _blocked_return_species(self, battle):
        battle_tag = getattr(battle, "battle_tag", None)
        if battle_tag is None:
            return None

        previous_species = self._last_switch_from.get(battle_tag)
        previous_turn = self._last_switch_turn.get(battle_tag)
        if previous_species is None or previous_turn is None:
            return None

        current_turn = int(getattr(battle, "turn", 0) or 0)
        if current_turn - previous_turn <= self.SWITCH_COOLDOWN_TURNS:
            return previous_species
        return None

    def _remember_voluntary_switch(self, battle, active):
        battle_tag = getattr(battle, "battle_tag", None)
        if battle_tag is None:
            return
        self._last_switch_from[battle_tag] = self._species(active)
        self._last_switch_turn[battle_tag] = int(
            getattr(battle, "turn", 0) or 0
        )

    @classmethod
    def best_switch(
        cls,
        battle: AbstractBattle,
        excluded_species=None,
    ):
        opponent = battle.opponent_active_pokemon
        switches = list(battle.available_switches or [])

        if excluded_species is not None:
            switches = [
                pokemon
                for pokemon in switches
                if str(getattr(pokemon, "species", ""))
                != excluded_species
            ]

        if not switches:
            return None, float("-inf")

        scored = [
            (cls.matchup_score(mon, opponent), mon)
            for mon in switches
        ]
        score, pokemon = max(scored, key=lambda item: item[0])
        return pokemon, float(score)

    def choose_move(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        current_matchup = self.matchup_score(active, opponent)

        # 1. Forced switch.
        if is_force_switch(battle):
            switch_target, switch_score = self.best_switch(battle)
            if switch_target is not None:
                order = self.create_order(switch_target)
                return self.record_order(
                    battle,
                    order,
                    heuristic_score=switch_score,
                    decision_reason="forced_best_matchup_switch",
                    current_matchup_score=current_matchup,
                    target_matchup_score=switch_score,
                )

        # 2. Voluntary strategic switch.
        blocked_species = self._blocked_return_species(battle)
        switch_target, switch_score = self.best_switch(
            battle,
            excluded_species=blocked_species,
        )

        improvement = (
            switch_score - current_matchup
            if switch_target is not None
            else float("-inf")
        )
        active_hp = float(hp_fraction(active) or 0.0)

        bad_matchup = current_matchup < 0.0
        meaningful_improvement = (
            improvement >= self.SWITCH_IMPROVEMENT_THRESHOLD
        )
        low_hp_escape = (
            active_hp <= self.LOW_HP_THRESHOLD
            and switch_target is not None
            and improvement >= 0.5
        )

        should_switch = (
            switch_target is not None
            and (
                (bad_matchup and meaningful_improvement)
                or low_hp_escape
            )
        )

        if should_switch:
            reason = (
                "bad_matchup_switch"
                if bad_matchup and meaningful_improvement
                else "low_hp_switch"
            )
            self._remember_voluntary_switch(battle, active)
            order = self.create_order(switch_target)
            return self.record_order(
                battle,
                order,
                heuristic_score=switch_score,
                decision_reason=reason,
                current_matchup_score=current_matchup,
                target_matchup_score=switch_score,
            )

        # 3. Best TypeAware attack.
        best_move, attack_score = self.best_scored_move(battle)
        if best_move is not None and attack_score > 0:
            order = self.create_order(best_move)
            return self.record_order(
                battle,
                order,
                heuristic_score=attack_score,
                decision_reason="best_type_aware_attack",
                current_matchup_score=current_matchup,
                target_matchup_score=(
                    switch_score if switch_target is not None else None
                ),
            )

        # 4. No positive offensive score: prefer a move, not a switch.
        moves = list(battle.available_moves or [])
        if moves:
            move = random.choice(moves)
            order = self.create_order(move)
            return self.record_order(
                battle,
                order,
                heuristic_score=0.0,
                decision_reason="fallback_move_no_positive_attack",
                current_matchup_score=current_matchup,
                target_matchup_score=(
                    switch_score if switch_target is not None else None
                ),
            )

        # 5. No move at all: switch if possible.
        if switch_target is not None:
            order = self.create_order(switch_target)
            return self.record_order(
                battle,
                order,
                heuristic_score=switch_score,
                decision_reason="no_move_available_switch",
                current_matchup_score=current_matchup,
                target_matchup_score=switch_score,
            )

        # 6. Defensive fallback.
        order, _ = self.choose_random_core_order(battle)
        return self.record_order(
            battle,
            order,
            heuristic_score=0.0,
            decision_reason="fallback_random_core_action",
            current_matchup_score=current_matchup,
        )
