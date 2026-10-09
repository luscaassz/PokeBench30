from poke_env.battle import AbstractBattle

from .base_agent import InstrumentedPlayer


class RandomAgent(InstrumentedPlayer):
    """Uniformly samples a valid action using poke-env's helper."""

    agent_name = "RandomAgent"

    def choose_move(self, battle: AbstractBattle):
        order = self.choose_random_move(battle)
        return self.record_order(battle, order)
