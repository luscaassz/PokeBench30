from __future__ import annotations

from typing import Any

import numpy as np

try:
    from gymnasium.spaces import Box, Discrete
    from poke_env.battle import AbstractBattle
    from poke_env.environment import SinglesEnv
    from poke_env.player import (
        BattleOrder,
        DefaultBattleOrder,
        ForfeitBattleOrder,
        Player,
        SingleBattleOrder,
    )
except ImportError as exc:  # pragma: no cover - helpful runtime error
    raise ImportError(
        "PokéBench30 RL environment requires poke-env and gymnasium. "
        "Run: python -m pip install -r requirements.txt"
    ) from exc

from pokebench30.representation.action_space import (
    ACTION_DIM,
    IllegalActionError,
    UniversalActionSpace,
)
from pokebench30.representation.encoder import UniversalStateEncoder
from pokebench30.representation.schema import build_feature_schema
from .reward import RewardConfig


class UniversalRLSinglesEnv(SinglesEnv):
    """poke-env SinglesEnv using a selectable PokéBench30 UBR profile.

    Core interface:
        UBR 1.1 -> observation: float32[472]
        UBR 1.2 -> observation: float32[518]
        action: Discrete(10)

    The actual PettingZoo/PokeEnv observation returned internally includes the
    normal poke-env dictionary with both ``observation`` and ``action_mask``.
    ``SB3MaskAdapter`` removes that transport wrapper for MaskablePPO.
    """

    def __init__(
        self,
        *,
        reward_config: RewardConfig | None = None,
        representation_version: str = "ubr-1.2",
        **kwargs: Any,
    ) -> None:
        self.representation_version = representation_version
        self.encoder = UniversalStateEncoder(
            representation_version=representation_version
        )
        self.reward_config = reward_config or RewardConfig()

        super().__init__(**kwargs)

        schema = build_feature_schema(representation_version)

        # SinglesEnv 0.16.x builds its native generation-dependent action
        # spaces explicitly. Override them with PokéBench30's 10-action space.
        self.action_spaces = {
            agent: Discrete(ACTION_DIM)
            for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: Box(
                low=-1.0,
                high=1.0,
                shape=(schema.observation_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

    @staticmethod
    def _is_wait_state(battle: Any) -> bool:
        return bool(
            getattr(battle, "wait", False)
            or getattr(battle, "_wait", False)
        )

    @staticmethod
    def get_action_space_size(gen: int) -> int:
        del gen
        return ACTION_DIM

    @staticmethod
    def get_action_mask(battle: Any) -> list[int]:
        # poke-env may expose a transport-only wait state. Keep one sentinel
        # action legal so masked policies never receive an all-zero mask.
        if UniversalRLSinglesEnv._is_wait_state(battle):
            return [1] + [0] * (ACTION_DIM - 1)

        return (
            UniversalActionSpace()
            .legal_mask(battle)
            .astype(np.int8)
            .tolist()
        )

    @staticmethod
    def action_to_order(
        action: np.int64 | int,
        battle: Any,
        fake: bool = False,
        strict: bool = True,
    ) -> BattleOrder:
        if UniversalRLSinglesEnv._is_wait_state(battle):
            return DefaultBattleOrder()

        try:
            action_index = int(action)
            selected = UniversalActionSpace().object_for_action(
                battle,
                action_index,
            )
            order = Player.create_order(selected)

            if (
                not fake
                and hasattr(battle, "valid_orders")
                and str(order) not in [str(o) for o in battle.valid_orders]
            ):
                raise ValueError(
                    f"UBR action {action_index} converted to {order}, "
                    "which is not a valid order for the current battle."
                )

            return order

        except (IllegalActionError, ValueError, IndexError) as exc:
            if strict:
                raise

            logger = getattr(battle, "logger", None)
            if logger is not None:
                logger.warning(
                    f"{exc} Defaulting to a random legal singles move."
                )
            return Player.choose_random_singles_move(battle)

    @staticmethod
    def order_to_action(
        order: BattleOrder,
        battle: Any,
        fake: bool = False,
        strict: bool = True,
    ) -> np.int64:
        try:
            if isinstance(order, DefaultBattleOrder):
                if UniversalRLSinglesEnv._is_wait_state(battle) or fake:
                    return np.int64(0)
                raise ValueError(
                    "DefaultBattleOrder is not a UBR core battle action."
                )

            if isinstance(order, ForfeitBattleOrder):
                raise ValueError(
                    "Forfeit is intentionally outside the UBR action space."
                )

            if not isinstance(order, SingleBattleOrder):
                raise ValueError(
                    f"Expected SingleBattleOrder, got {type(order).__name__}."
                )

            if (
                not fake
                and hasattr(battle, "valid_orders")
                and str(order) not in [str(o) for o in battle.valid_orders]
            ):
                raise ValueError(
                    f"Order {order} is not valid for the current battle."
                )

            action = UniversalActionSpace().index_for_order(
                battle,
                order,
            )
            if action is None:
                raise ValueError(
                    f"Order {order} cannot be represented by the selected UBR profile."
                )

            return np.int64(action)

        except ValueError:
            if strict:
                raise

            mask = UniversalRLSinglesEnv.get_action_mask(battle)
            legal = [i for i, value in enumerate(mask) if value]
            return np.int64(legal[0] if legal else 0)

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return self.encoder.encode(battle).vector

    def calc_reward(self, battle: AbstractBattle) -> float:
        cfg = self.reward_config
        return self.reward_computing_helper(
            battle,
            fainted_value=cfg.fainted_value,
            hp_value=cfg.hp_value,
            status_value=cfg.status_value,
            victory_value=cfg.victory_value,
        )
