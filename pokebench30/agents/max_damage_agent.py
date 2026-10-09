from poke_env.battle import AbstractBattle

from .base_agent import InstrumentedPlayer


class MaxDamageAgent(InstrumentedPlayer):
    """Selects the available damaging move with the highest Base Power."""

    agent_name = "MaxDamageAgent"

    def choose_move(self, battle: AbstractBattle):
        if battle.available_moves:
            best_move = max(
                battle.available_moves,
                key=lambda move: float(move.base_power or 0),
            )
            if float(best_move.base_power or 0) > 0:
                order = self.create_order(best_move)
                return self.record_order(
                    battle,
                    order,
                    heuristic_score=float(best_move.base_power or 0),
                    decision_reason="max_base_power",
                )

        order, _ = self.choose_random_core_order(battle)
        return self.record_order(
            battle,
            order,
            heuristic_score=0.0,
            decision_reason="fallback_random_core_action",
        )
