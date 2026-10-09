from poke_env.battle import AbstractBattle

from .base_agent import InstrumentedPlayer


class MaxDamageAgent(InstrumentedPlayer):
    """Selects the available move with the highest Base Power."""

    agent_name = "MaxDamageAgent"

    def choose_move(self, battle: AbstractBattle):
        if battle.available_moves:
            best_move = max(
                battle.available_moves,
                key=lambda move: float(move.base_power or 0),
            )

            # Intentionally does not activate generation-specific special mechanics.
            order = self.create_order(best_move)
            return self.record_order(
                battle,
                order,
                score=float(best_move.base_power or 0),
            )

        order = self.choose_random_move(battle)
        return self.record_order(battle, order)
