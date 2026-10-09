from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from poke_env.battle import AbstractBattle
from poke_env.player import DefaultBattleOrder

from pokebench30.agents.base_agent import InstrumentedPlayer
from pokebench30.representation.action_space import IllegalActionError
from pokebench30.representation.encoder import UniversalStateEncoder
from .model_io import load_model_metadata, resolve_model_file, validate_model_metadata


class RLPolicyAgent(InstrumentedPlayer):
    """Inference-only MaskablePPO player using the model's recorded UBR profile."""

    agent_name = "RLPolicyAgent"

    def __init__(
        self,
        *args: Any,
        model_path: str | Path,
        deterministic: bool = True,
        require_same_generation: bool = True,
        **kwargs: Any,
    ) -> None:
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise ImportError(
                "RLPolicyAgent requires sb3-contrib. "
                "Run: python -m pip install -r requirements.txt"
            ) from exc

        self.model_path = resolve_model_file(model_path)
        self.model_metadata = load_model_metadata(self.model_path)
        self.deterministic = bool(deterministic)
        self.require_same_generation = bool(require_same_generation)
        state_encoder = UniversalStateEncoder(
            representation_version=self.model_metadata.observation_schema_version
        )

        super().__init__(
            *args,
            state_encoder=state_encoder,
            **kwargs,
        )

        self.model = MaskablePPO.load(
            str(self.model_path),
            device="auto",
        )

    @staticmethod
    def _generation_from_battle(battle: AbstractBattle) -> int:
        try:
            return int(getattr(battle, "gen"))
        except (TypeError, ValueError, AttributeError):
            return 0

    def choose_move(self, battle: AbstractBattle):
        if bool(
            getattr(battle, "wait", False)
            or getattr(battle, "_wait", False)
        ):
            return DefaultBattleOrder()

        generation = self._generation_from_battle(battle)
        validate_model_metadata(
            self.model_metadata,
            generation=generation if generation else None,
            require_same_generation=self.require_same_generation,
        )

        encoded = self.state_encoder.encode(battle)
        mask = encoded.legal_action_mask.astype(bool)

        if not np.any(mask):
            order, _ = self.choose_random_core_order(battle)
            return self.record_order(
                battle,
                order,
                decision_reason="rl_zero_mask_fallback",
            )

        action, _ = self.model.predict(
            encoded.vector,
            action_masks=mask,
            deterministic=self.deterministic,
        )
        action_index = int(np.asarray(action).item())

        try:
            selected = self.state_encoder.action_space.object_for_action(
                battle,
                action_index,
            )
        except IllegalActionError:
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                order, _ = self.choose_random_core_order(battle)
                return self.record_order(
                    battle,
                    order,
                    decision_reason="rl_illegal_action_no_legal_fallback",
                )
            action_index = int(legal[0])
            selected = self.state_encoder.action_space.object_for_action(
                battle,
                action_index,
            )
            reason = "rl_illegal_action_first_legal_fallback"
        else:
            reason = "maskable_ppo_policy"

        order = self.create_order(selected)

        return self.record_order(
            battle,
            order,
            decision_reason=reason,
        )
