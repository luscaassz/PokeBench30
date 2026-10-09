import tempfile
import unittest
from pathlib import Path
import json

from pokebench30.representation.schema import build_feature_schema
from pokebench30.rl.model_io import (
    ModelMetadata,
    load_model_metadata,
    resolve_model_file,
    validate_model_metadata,
)
from pokebench30.rl.reward import RewardConfig


class RLModelIOTests(unittest.TestCase):
    def make_metadata(self, generation=9, representation="ubr-1.2"):
        schema = build_feature_schema(representation)
        return ModelMetadata(
            project_version="0.5.2",
            algorithm="MaskablePPO",
            training_generation=generation,
            training_battle_format=f"gen{generation}randombattle",
            training_opponent="type_aware",
            total_timesteps=1000,
            seed=42,
            observation_schema_version=schema.version,
            observation_schema_sha256=schema.sha256,
            observation_dim=schema.observation_dim,
            action_dim=schema.action_dim,
            reward=RewardConfig().to_dict(),
            library_versions={},
            notes=[],
        )

    def test_ubr12_is_accepted(self):
        validate_model_metadata(self.make_metadata(9, "ubr-1.2"), generation=9)

    def test_ubr11_is_accepted(self):
        validate_model_metadata(self.make_metadata(9, "ubr-1.1"), generation=9)

    def test_expected_profile_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_model_metadata(
                self.make_metadata(9, "ubr-1.1"),
                generation=9,
                expected_representation_version="ubr-1.2",
            )

    def test_cross_generation_is_rejected_in_v05(self):
        with self.assertRaises(ValueError):
            validate_model_metadata(
                self.make_metadata(9),
                generation=4,
                require_same_generation=True,
            )

    def test_metadata_round_trip(self):
        metadata = self.make_metadata(9, "ubr-1.1")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata.json").write_text(
                json.dumps(metadata.to_dict()), encoding="utf-8"
            )
            loaded = load_model_metadata(root / "maskable_ppo.zip")
        self.assertEqual(loaded.observation_schema_sha256, metadata.observation_schema_sha256)
        self.assertEqual(loaded.observation_schema_version, "ubr-1.1")

    def test_resolve_model_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "maskable_ppo.zip"
            model.write_bytes(b"placeholder")
            self.assertEqual(resolve_model_file(root), model.resolve())

    def test_reward_defaults(self):
        reward = RewardConfig()
        self.assertEqual(reward.victory_value, 30.0)
        self.assertEqual(reward.fainted_value, 2.0)
        self.assertEqual(reward.hp_value, 1.0)
        self.assertEqual(reward.status_value, 0.5)


    def test_policy_metadata_defaults_are_backward_compatible(self):
        metadata = self.make_metadata(9, "ubr-1.2")
        self.assertEqual(metadata.policy_architecture, "mlp_baseline")
        self.assertEqual(metadata.policy_config, {})
        self.assertIsNone(metadata.model_input_dim)

    def test_temporal_input_dimension_validation(self):
        metadata = self.make_metadata(9, "ubr-1.2")
        metadata = ModelMetadata(
            **{
                **metadata.to_dict(),
                "policy_architecture": "history_lstm",
                "policy_config": {"history_len": 4},
                "model_input_dim": 2072,
            }
        )
        validate_model_metadata(metadata, generation=9)


if __name__ == "__main__":
    unittest.main()
