from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from pokebench30.representation.schema import build_feature_schema


@dataclass(frozen=True)
class ModelMetadata:
    project_version: str
    algorithm: str
    training_generation: int
    training_battle_format: str
    training_opponent: str
    total_timesteps: int
    seed: int
    observation_schema_version: str
    observation_schema_sha256: str
    observation_dim: int
    action_dim: int
    reward: dict[str, float]
    library_versions: dict[str, str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_model_file(model_path: str | Path) -> Path:
    path = Path(model_path).resolve()

    if path.is_dir():
        candidate = path / "maskable_ppo.zip"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Could not find {candidate}. Expected a v0.5 model directory "
                "containing maskable_ppo.zip."
            )
        return candidate

    if path.exists():
        return path

    if path.suffix != ".zip":
        candidate = path.with_suffix(".zip")
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"RL model not found: {path}")


def model_metadata_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.is_dir():
        return path / "metadata.json"
    if path.suffix == ".zip":
        return path.parent / "metadata.json"
    return path.parent / "metadata.json"


def load_model_metadata(model_path: str | Path) -> ModelMetadata:
    path = model_metadata_path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at {path}. "
            "PokéBench30 v0.5 models must be distributed with metadata.json."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ModelMetadata(**raw)


def validate_model_metadata(
    metadata: ModelMetadata,
    *,
    generation: int | None = None,
    require_same_generation: bool = True,
) -> None:
    schema = build_feature_schema()

    errors: list[str] = []

    if metadata.observation_schema_version != schema.version:
        errors.append(
            "schema version mismatch: "
            f"model={metadata.observation_schema_version}, current={schema.version}"
        )

    if metadata.observation_schema_sha256 != schema.sha256:
        errors.append(
            "schema SHA-256 mismatch: "
            f"model={metadata.observation_schema_sha256}, current={schema.sha256}"
        )

    if metadata.observation_dim != schema.observation_dim:
        errors.append(
            "observation dimension mismatch: "
            f"model={metadata.observation_dim}, current={schema.observation_dim}"
        )

    if metadata.action_dim != schema.action_dim:
        errors.append(
            "action dimension mismatch: "
            f"model={metadata.action_dim}, current={schema.action_dim}"
        )

    if (
        generation is not None
        and require_same_generation
        and int(metadata.training_generation) != int(generation)
    ):
        errors.append(
            "generation mismatch: "
            f"model trained on Gen {metadata.training_generation}, "
            f"requested Gen {generation}. "
            "Cross-generation evaluation is intentionally reserved for v0.6."
        )

    if errors:
        raise ValueError("Incompatible RL model:\n- " + "\n- ".join(errors))
