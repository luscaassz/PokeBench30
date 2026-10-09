from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .encoder import EncodedObservation
from .schema import (
    MAX_TEAM_SIZE,
    SIDE_CONDITION_FEATURES,
    build_feature_schema,
)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class ObservationValidator:
    def __init__(self) -> None:
        self.schema = build_feature_schema()
        self._indices = {
            name: index
            for index, name in enumerate(
                self.schema.feature_names
            )
        }

    def validate(
        self,
        observation: EncodedObservation,
        battle: Any | None = None,
    ) -> ValidationReport:
        report = ValidationReport()

        vector = np.asarray(observation.vector)
        mask = np.asarray(observation.legal_action_mask)

        if vector.shape != (self.schema.observation_dim,):
            report.errors.append(
                f"Wrong observation shape: {vector.shape}; "
                f"expected {(self.schema.observation_dim,)}."
            )

        if mask.shape != (self.schema.action_dim,):
            report.errors.append(
                f"Wrong action-mask shape: {mask.shape}; "
                f"expected {(self.schema.action_dim,)}."
            )

        if not np.all(np.isfinite(vector)):
            report.errors.append(
                "Observation contains NaN or infinity."
            )

        if vector.size and (
            np.min(vector) < -1.000001
            or np.max(vector) > 1.000001
        ):
            report.errors.append(
                "Observation contains value outside [-1, 1]."
            )

        if not np.all(np.isin(mask, [0, 1])):
            report.errors.append(
                "Legal-action mask is not binary."
            )

        if observation.schema_sha256 != self.schema.sha256:
            report.errors.append(
                "Observation schema hash does not match the "
                "current encoder schema."
            )

        # Unknown opponent slots must be true zero-padding. This ensures
        # hidden team members cannot leave feature traces.
        for slot in range(1, MAX_TEAM_SIZE + 1):
            prefix = f"opponent_team.slot_{slot}"
            known_index = self._indices[f"{prefix}.known"]

            if known_index >= vector.size:
                continue

            if float(vector[known_index]) == 0.0:
                slot_indices = [
                    index
                    for name, index in self._indices.items()
                    if name.startswith(prefix + ".")
                ]
                if any(
                    abs(float(vector[index])) > 1e-8
                    for index in slot_indices
                ):
                    report.errors.append(
                        f"Unknown opponent slot {slot} is not "
                        "zero-padded."
                    )

        # UBR 1.2 battle context is binary except normalized hazard layers.
        continuous_context_suffixes = {
            ".spikes",
            ".toxic_spikes",
        }
        for name, index in self._indices.items():
            if not name.startswith("battle_context."):
                continue
            if index >= vector.size:
                continue
            if any(
                name.endswith(suffix)
                for suffix in continuous_context_suffixes
            ):
                continue

            value = float(vector[index])
            if not (
                abs(value - 0.0) <= 1e-8
                or abs(value - 1.0) <= 1e-8
            ):
                report.errors.append(
                    f"Battle-context feature {name} is not binary: "
                    f"{value}."
                )

        if battle is not None:
            force_switch = getattr(
                battle,
                "force_switch",
                False,
            )
            if isinstance(force_switch, (list, tuple)):
                force_switch = any(bool(v) for v in force_switch)
            else:
                force_switch = bool(force_switch)

            if force_switch and np.any(mask[:4] != 0):
                report.errors.append(
                    "Move action is legal while a forced switch "
                    "is required."
                )

            opponent_team = getattr(
                battle,
                "opponent_team",
                {},
            ) or {}
            actual_revealed = len(
                opponent_team.values()
                if hasattr(opponent_team, "values")
                else opponent_team
            )
            encoded_revealed = int(
                observation.metadata.get(
                    "opponent_revealed",
                    -1,
                )
            )

            if encoded_revealed != actual_revealed:
                # The active opponent may be inserted by the action-space
                # adapter if poke-env has not yet inserted it in the mapping.
                if not (
                    encoded_revealed == actual_revealed + 1
                    and getattr(
                        battle,
                        "opponent_active_pokemon",
                        None,
                    ) is not None
                ):
                    report.errors.append(
                        "Encoded opponent reveal count does not "
                        "match the agent-visible battle state."
                    )

        if mask.size and int(mask.sum()) == 0:
            report.warnings.append(
                "No legal core action is present in the mask. "
                "This can occur only in unusual terminal/request states."
            )

        return report
