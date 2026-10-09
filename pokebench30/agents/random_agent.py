from __future__ import annotations

from poke_env.battle import AbstractBattle

from .base_agent import InstrumentedPlayer


class RandomAgent(InstrumentedPlayer):
    """Uniform random baseline over ordinary legal moves and switches."""

    agent_name = "RandomAgent"

    def choose_move(self, battle: AbstractBattle):
        order, action_type = self.choose_random_core_order(battle)
        reason = {
            "move": "uniform_random_core_move",
            "switch": "uniform_random_core_switch",
        }.get(action_type, "fallback_random_legal_action")

        return self.record_order(
            battle,
            order,
            decision_reason=reason,
        )
