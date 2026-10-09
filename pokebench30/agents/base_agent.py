from __future__ import annotations

from collections import defaultdict
from typing import Any

from poke_env.player import Player


class InstrumentedPlayer(Player):
    """Base player with lightweight decision logging.

    The battle outcome itself is read from ``Player.battles`` after the battle.
    This class only records the decisions requested from the agent.
    """

    agent_name = "InstrumentedPlayer"

    def __init__(self, *args: Any, log_decisions: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_decisions = log_decisions
        self.decision_log: list[dict[str, Any]] = []
        self.decision_counts: dict[str, int] = defaultdict(int)

    @staticmethod
    def _action_info(order: Any) -> tuple[str, str]:
        """Best-effort classification of a poke-env battle order."""
        selected = getattr(order, "order", None)

        if selected is None:
            return "other", order.__class__.__name__

        # Move-like objects expose id/base_power; Pokemon-like objects expose species.
        move_id = getattr(selected, "id", None)
        if move_id is not None:
            return "move", str(move_id)

        species = getattr(selected, "species", None)
        if species is not None:
            return "switch", str(species)

        return "other", str(selected)

    def record_order(
        self,
        battle: Any,
        order: Any,
        *,
        score: float | None = None,
    ) -> Any:
        battle_tag = getattr(battle, "battle_tag", "unknown")
        self.decision_counts[battle_tag] += 1

        if self.log_decisions:
            action_type, action = self._action_info(order)
            self.decision_log.append(
                {
                    "battle_tag": battle_tag,
                    "turn": getattr(battle, "turn", None),
                    "agent": self.agent_name,
                    "action_type": action_type,
                    "action": action,
                    "score": score,
                }
            )

        return order
