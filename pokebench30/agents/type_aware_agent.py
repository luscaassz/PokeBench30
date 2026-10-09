from __future__ import annotations

from poke_env.battle import AbstractBattle

from .base_agent import InstrumentedPlayer


class TypeAwareAgent(InstrumentedPlayer):
    """Simple interpretable offensive heuristic.

    score = base_power × accuracy × STAB × type_effectiveness

    This is NOT the Pokémon damage formula. It is intentionally a compact
    baseline that uses information available in the battle state.
    """

    agent_name = "TypeAwareAgent"

    @staticmethod
    def _accuracy_fraction(accuracy) -> float:
        # poke-env commonly represents move accuracy as a fraction.
        # This defensive conversion also tolerates percentages.
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
    def score_move(cls, battle: AbstractBattle, move) -> float:
        base_power = float(move.base_power or 0)
        if base_power <= 0:
            return 0.0

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
            # poke-env applies the type chart for the Pokémon's generation.
            effectiveness = float(opponent.damage_multiplier(move))

        return base_power * accuracy * stab * effectiveness

    def choose_move(self, battle: AbstractBattle):
        if battle.available_moves:
            scored_moves = [
                (self.score_move(battle, move), move)
                for move in battle.available_moves
            ]
            score, best_move = max(scored_moves, key=lambda item: item[0])

            # If every move is non-damaging, use any valid action instead of
            # deterministically selecting the first status move.
            if score <= 0:
                order = self.choose_random_move(battle)
                return self.record_order(battle, order, score=0.0)

            order = self.create_order(best_move)
            return self.record_order(battle, order, score=score)

        order = self.choose_random_move(battle)
        return self.record_order(battle, order)
