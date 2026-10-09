from __future__ import annotations

import random

from poke_env.battle import AbstractBattle

from .base_agent import InstrumentedPlayer, is_force_switch


class TypeAwareAgent(InstrumentedPlayer):
    """Interpretable offensive heuristic.

    score = base_power × accuracy × STAB × type_effectiveness

    TypeAwareAgent never performs a voluntary switch. This keeps it as the
    clean attack-only comparator for SwitchAwareAgent.
    """

    agent_name = "TypeAwareAgent"

    @staticmethod
    def _accuracy_fraction(accuracy) -> float:
        if accuracy is None or accuracy is True:
            return 1.0
        try:
            value = float(accuracy)
        except (TypeError, ValueError):
            return 1.0
        if value <= 0:
            return 1.0
        if value > 1:
            return min(value / 100.0, 1.0)
        return value

    @classmethod
    def score_move_components(cls, battle: AbstractBattle, move) -> dict[str, float]:
        base_power = float(move.base_power or 0)
        accuracy = cls._accuracy_fraction(getattr(move, "accuracy", 1.0))

        stab = 1.0
        active = battle.active_pokemon
        if active is not None:
            active_types = {
                type_
                for type_ in (
                    getattr(active, "type_1", None),
                    getattr(active, "type_2", None),
                )
                if type_ is not None
            }
            if getattr(move, "type", None) in active_types:
                stab = 1.5

        effectiveness = 1.0
        opponent = battle.opponent_active_pokemon
        if opponent is not None:
            try:
                effectiveness = float(opponent.damage_multiplier(move))
            except (AttributeError, TypeError, ValueError):
                effectiveness = 1.0

        score = base_power * accuracy * stab * effectiveness
        return {
            "base_power": base_power,
            "accuracy": accuracy,
            "stab": stab,
            "effectiveness": effectiveness,
            "score": score,
        }

    @classmethod
    def score_move(cls, battle: AbstractBattle, move) -> float:
        return cls.score_move_components(battle, move)["score"]

    @classmethod
    def best_scored_move(cls, battle: AbstractBattle):
        if not battle.available_moves:
            return None, 0.0
        scored = [
            (cls.score_move(battle, move), move)
            for move in battle.available_moves
        ]
        score, move = max(scored, key=lambda item: item[0])
        return move, float(score)

    def choose_move(self, battle: AbstractBattle):
        # Forced switches are allowed; voluntary switches are not.
        if is_force_switch(battle):
            switches = list(battle.available_switches or [])
            if switches:
                target = random.choice(switches)
                order = self.create_order(target)
                return self.record_order(
                    battle,
                    order,
                    heuristic_score=0.0,
                    decision_reason="forced_random_switch",
                )

        best_move, score = self.best_scored_move(battle)
        if best_move is not None and score > 0:
            order = self.create_order(best_move)
            return self.record_order(
                battle,
                order,
                heuristic_score=score,
                decision_reason="best_type_aware_attack",
            )

        # If no damaging move has positive score, still choose a move instead
        # of voluntarily switching.
        moves = list(battle.available_moves or [])
        if moves:
            move = random.choice(moves)
            order = self.create_order(move)
            return self.record_order(
                battle,
                order,
                heuristic_score=0.0,
                decision_reason="fallback_random_move",
            )

        # Defensive fallback for unusual states with no move available.
        switches = list(battle.available_switches or [])
        if switches:
            target = random.choice(switches)
            order = self.create_order(target)
            return self.record_order(
                battle,
                order,
                heuristic_score=0.0,
                decision_reason="fallback_required_switch",
            )

        order, _ = self.choose_random_core_order(battle)
        return self.record_order(
            battle,
            order,
            heuristic_score=0.0,
            decision_reason="fallback_random_core_action",
        )
