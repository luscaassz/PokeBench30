import unittest

from pokebench30.representation.schema import build_feature_schema


class RepresentationSchemaTests(unittest.TestCase):
    def test_schema_shape_is_fixed(self):
        schema = build_feature_schema()

        self.assertEqual(schema.observation_dim, 472)
        self.assertEqual(schema.action_dim, 10)
        self.assertEqual(
            len(schema.feature_names),
            len(set(schema.feature_names)),
        )
        self.assertEqual(
            len(schema.action_labels),
            len(set(schema.action_labels)),
        )

    def test_schema_hash_is_stable_for_same_schema(self):
        a = build_feature_schema()
        b = build_feature_schema()

        self.assertEqual(a.sha256, b.sha256)


if __name__ == "__main__":
    unittest.main()
