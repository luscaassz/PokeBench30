import unittest

from pokebench30.representation.schema import build_feature_schema


class RepresentationSchemaTests(unittest.TestCase):
    def test_both_schema_profiles_are_stable(self):
        ubr11 = build_feature_schema("ubr-1.1")
        ubr12 = build_feature_schema("ubr-1.2")

        self.assertEqual(ubr11.version, "ubr-1.1")
        self.assertEqual(ubr11.observation_dim, 472)
        self.assertEqual(ubr11.action_dim, 10)
        self.assertEqual(
            ubr11.sha256,
            "276327ddc0883116305f6d564cb34ba43d7cf2eff8294185aeb9587c4485a20e",
        )

        self.assertEqual(ubr12.version, "ubr-1.2")
        self.assertEqual(ubr12.observation_dim, 518)
        self.assertEqual(ubr12.action_dim, 10)
        self.assertEqual(
            ubr12.sha256,
            "d4b3858aab3a1c3bb1e17d9f34becf93814be05bfc949d52023651035f20e4c3",
        )

        self.assertEqual(
            ubr12.feature_names[: ubr11.observation_dim],
            ubr11.feature_names,
        )
        self.assertEqual(
            ubr12.feature_names[472],
            "battle_context.own_side.stealth_rock",
        )

    def test_default_profile_remains_ubr12(self):
        schema = build_feature_schema()
        self.assertEqual(schema.version, "ubr-1.2")
        self.assertEqual(schema.observation_dim, 518)

    def test_invalid_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            build_feature_schema("ubr-9.9")


if __name__ == "__main__":
    unittest.main()
