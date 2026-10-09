from __future__ import annotations

from poke_env.battle import AbstractBattle

from .base_agent import hp_fraction, is_force_switch
from .type_aware_agent import TypeAwareAgent


class SwitchAwareAgent(TypeAwareAgent):
    """Type-aware attacker that can voluntarily switch out of bad matchups.

    The switch heuristic is intentionally transparent rather than optimal.
    It compares a simple type matchup score for the active Pokémon and each
    legal switch target.
    """

    agent_name = "SwitchAwareAgent"
    SWITCH_IMPROVEMENT_THRESHOLD = 1.0
    LOW_HP_THRESHOLD = 0.20

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

    @classmethod
    def best_switch(cls, battle: AbstractBattle):
        opponent = battle.opponent_active_pokemon
        switches = list(battle.available_switches or [])
        if not switches:
            return None, float("-inf")

        scored = [(cls.matchup_score(mon, opponent), mon) for mon in switches]
        score, pokemon = max(scored, key=lambda item: item[0])
        return pokemon, float(score)

    def choose_move(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        current_matchup = self.matchup_score(active, opponent)
        switch_target, switch_score = self.best_switch(battle)

        if is_force_switch(battle) and switch_target is not None:
            order = self.create_order(switch_target)
            return self.record_order(
                battle,
                order,
                heuristic_score=switch_score,
                decision_reason="forced_best_matchup_switch",
                current_matchup_score=current_matchup,
                target_matchup_score=switch_score,
            )

        improvement = switch_score - current_matchup if switch_target is not None else float("-inf")
        active_hp = float(hp_fraction(active) or 0.0)

        bad_matchup = current_matchup < 0.0
        meaningful_improvement = improvement >= self.SWITCH_IMPROVEMENT_THRESHOLD
        low_hp_escape = (
            active_hp <= self.LOW_HP_THRESHOLD
            and switch_target is not None
            and switch_score > current_matchup + 0.5
        )

        if switch_target is not None and ((bad_matchup and meaningful_improvement) or low_hp_escape):
            reason = "bad_matchup_switch" if bad_matchup and meaningful_improvement else "low_hp_switch"
            order = self.create_order(switch_target)
            return self.record_order(
                battle,
                order,
                heuristic_score=switch_score,
                decision_reason=reason,
                current_matchup_score=current_matchup,
                target_matchup_score=switch_score,
            )

        best_move, attack_score = self.best_scored_move(battle)
        if best_move is not None and attack_score > 0:
            order = self.create_order(best_move)
            return self.record_order(
                battle,
                order,
                heuristic_score=attack_score,
                decision_reason="best_type_aware_attack",
                current_matchup_score=current_matchup,
                target_matchup_score=switch_score if switch_target is not None else None,
            )

        if switch_target is not None:
            order = self.create_order(switch_target)
            return self.record_order(
                battle,
                order,
                heuristic_score=switch_score,
                decision_reason="no_positive_attack_switch",
                current_matchup_score=current_matchup,
                target_matchup_score=switch_score,
            )

        order = self.choose_random_move(battle)
        return self.record_order(
            battle,
            order,
            heuristic_score=0.0,
            decision_reason="fallback_random_legal_action",
            current_matchup_score=current_matchup,
        )
