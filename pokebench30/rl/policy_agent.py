from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from poke_env.battle import AbstractBattle
from poke_env.player import DefaultBattleOrder

from pokebench30.agents.base_agent import InstrumentedPlayer
from pokebench30.representation.action_space import IllegalActionError
from pokebench30.representation.encoder import UniversalStateEncoder
from .model_io import load_model_metadata, resolve_model_file, validate_model_metadata
from .policy_specs import get_policy_spec, padded_history


class RLPolicyAgent(InstrumentedPlayer):
    """Inference-only RL player for MaskablePPO and masked Q-learning models."""

    agent_name = "RLPolicyAgent"

    def __init__(
        self,
        *args: Any,
        model_path: str | Path,
        deterministic: bool = True,
        require_same_generation: bool = True,
        **kwargs: Any,
    ) -> None:
        # Import custom extractor before deserializing temporal PPO models.
        try:
            from . import temporal_policy as _temporal_policy  # noqa: F401
        except ImportError:
            _temporal_policy = None

        try:
            from . import wang_style as _wang_style  # noqa: F401
        except ImportError:
            _wang_style = None

        self.model_path = resolve_model_file(model_path)
        self.model_metadata = load_model_metadata(self.model_path)
        self.deterministic = bool(deterministic)
        self.require_same_generation = bool(require_same_generation)

        architecture = getattr(
            self.model_metadata,
            "policy_architecture",
            "mlp_baseline",
        ) or "mlp_baseline"
        self.policy_spec = get_policy_spec(architecture)

        state_encoder = UniversalStateEncoder(
            representation_version=self.model_metadata.observation_schema_version
        )
        self._base_dim = state_encoder.schema.observation_dim
        self._history: dict[str, deque[np.ndarray]] = {}
        self._last_vector: dict[str, np.ndarray] = {}

        super().__init__(
            *args,
            state_encoder=state_encoder,
            **kwargs,
        )

        if self.model_metadata.algorithm == "MaskablePPO":
            try:
                from sb3_contrib import MaskablePPO
            except ImportError as exc:
                raise ImportError(
                    "MaskablePPO inference requires sb3-contrib. "
                    "Run: python -m pip install -r requirements.txt"
                ) from exc

            self.model = MaskablePPO.load(
                str(self.model_path),
                device="auto",
            )
            self._value_based = False
            self._dict_mask_policy = False

        elif self.model_metadata.algorithm == "PPO":
            try:
                from stable_baselines3 import PPO
            except ImportError as exc:
                raise ImportError(
                    "PPO inference requires stable-baselines3. "
                    "Run: python -m pip install -r requirements.txt"
                ) from exc

            # Import custom v0.6 policy class before deserialization.
            from . import wang_style as _wang_style  # noqa: F401

            self.model = PPO.load(
                str(self.model_path),
                device="auto",
            )
            self._value_based = False
            self._dict_mask_policy = True

        elif self.model_metadata.algorithm in {
            "MaskedDQN",
            "MaskedDoubleDQN",
        }:
            from .masked_dqn import MaskedQPolicy

            self.model = MaskedQPolicy.load(
                self.model_path,
                device="auto",
            )
            self._value_based = True
            self._dict_mask_policy = False

        else:
            raise ValueError(
                f"Unsupported RL algorithm in metadata: "
                f"{self.model_metadata.algorithm!r}"
            )

    @staticmethod
    def _generation_from_battle(battle: AbstractBattle) -> int:
        try:
            return int(getattr(battle, "gen"))
        except (TypeError, ValueError, AttributeError):
            return 0

    @staticmethod
    def _battle_key(battle: AbstractBattle) -> str:
        return str(
            getattr(battle, "battle_tag", None)
            or getattr(battle, "_battle_tag", None)
            or id(battle)
        )

    def _model_observation(
        self,
        battle: AbstractBattle,
        vector: np.ndarray,
    ) -> np.ndarray:
        if self.policy_spec.history_len <= 1:
            return np.asarray(vector, dtype=np.float32)

        key = self._battle_key(battle)
        if key not in self._history:
            self._history[key] = deque(maxlen=self.policy_spec.history_len)
            self._last_vector.pop(key, None)

        current = np.asarray(vector, dtype=np.float32).reshape(-1)
        previous = self._last_vector.get(key)

        # Avoid duplicating identical transport requests, while still allowing
        # multiple distinct decisions within the same battle turn.
        if previous is None or not np.array_equal(previous, current):
            self._history[key].append(current.copy())
            self._last_vector[key] = current.copy()

        return padded_history(
            self._history[key],
            history_len=self.policy_spec.history_len,
            base_dim=self._base_dim,
        )

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

        model_obs = self._model_observation(battle, encoded.vector)
        if self._value_based:
            action_index = int(
                self.model.predict(
                    model_obs,
                    action_mask=mask,
                )
            )
        elif self._dict_mask_policy:
            action, _ = self.model.predict(
                {
                    "observation": np.asarray(
                        model_obs,
                        dtype=np.float32,
                    ),
                    "action_mask": mask.astype(np.int8),
                },
                deterministic=self.deterministic,
            )
            action_index = int(np.asarray(action).item())
        else:
            action, _ = self.model.predict(
                model_obs,
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
            if self._value_based:
                reason = (
                    f"{self.model_metadata.algorithm.lower()}_policy:"
                    f"{self.policy_spec.name}"
                )
            elif self._dict_mask_policy:
                reason = f"ppo_masked_policy:{self.policy_spec.name}"
            else:
                reason = f"maskable_ppo_policy:{self.policy_spec.name}"

        order = self.create_order(selected)

        return self.record_order(
            battle,
            order,
            decision_reason=reason,
        )
