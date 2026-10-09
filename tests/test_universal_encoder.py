import unittest

import numpy as np

from pokebench30.representation.encoder import UniversalStateEncoder
from pokebench30.representation.validator import ObservationValidator


class EnumLike:
    def __init__(self, name):
        self.name = name


class DummyMove:
    def __init__(
        self,
        move_id,
        type_name="NORMAL",
        category="PHYSICAL",
        base_power=80,
        accuracy=1.0,
        priority=0,
    ):
        self.id = move_id
        self.type = EnumLike(type_name)
        self.category = EnumLike(category)
        self.base_power = base_power
        self.accuracy = accuracy
        self.priority = priority


class DummyPokemon:
    def __init__(
        self,
        species,
        type_1="NORMAL",
        type_2=None,
        hp=1.0,
        active=False,
        fainted=False,
        status=None,
        moves=None,
        level=100,
    ):
        self.species = species
        self.name = species
        self.type_1 = EnumLike(type_1)
        self.type_2 = EnumLike(type_2) if type_2 else None
        self.current_hp_fraction = hp
        self.active = active
        self.fainted = fainted
        self.status = EnumLike(status) if status else None
        self.level = level
        self.boosts = {}
        self.moves = {
            move.id: move for move in (moves or [])
        }


class HiddenTrap:
    def __getattribute__(self, name):
        raise AssertionError(
            "Encoder attempted to access hidden ground truth."
        )


class DummyBattle:
    def __init__(self, gen):
        move = DummyMove(
            "surf",
            type_name="WATER",
            category="SPECIAL",
            base_power=90,
        )
        self.gen = gen
        self.turn = 12
        self.battle_tag = f"dummy-gen{gen}"
        self.force_switch = False

        self.active_pokemon = DummyPokemon(
            "own-a",
            type_1="WATER",
            active=True,
            moves=[move],
        )
        self.own_b = DummyPokemon(
            "own-b",
            type_1="GRASS",
        )
        self.team = {
            "own-a": self.active_pokemon,
            "own-b": self.own_b,
        }

        self.opponent_active_pokemon = DummyPokemon(
            "seen-a",
            type_1="FIRE",
            hp=0.5,
            active=True,
        )
        self.opponent_team = {
            "seen-a": self.opponent_active_pokemon,
        }

        # A deliberately hostile attribute. UBR must never use it.
        self.opponent_full_team = HiddenTrap()

        self.available_moves = [move]
        self.available_switches = [self.own_b]


class UniversalEncoderTests(unittest.TestCase):
    def test_same_shape_across_generations(self):
        encoder = UniversalStateEncoder()

        shapes = {
            encoder.encode(DummyBattle(gen)).vector.shape
            for gen in (1, 4, 9)
        }

        self.assertEqual(shapes, {(472,)})

    def test_encoding_is_deterministic(self):
        encoder = UniversalStateEncoder()
        battle = DummyBattle(9)

        first = encoder.encode(battle)
        second = encoder.encode(battle)

        np.testing.assert_array_equal(
            first.vector,
            second.vector,
        )
        np.testing.assert_array_equal(
            first.legal_action_mask,
            second.legal_action_mask,
        )

    def test_hidden_opponent_slots_are_zero(self):
        encoder = UniversalStateEncoder()
        observation = encoder.encode(DummyBattle(9))
        validator = ObservationValidator()

        report = validator.validate(observation)

        self.assertTrue(report.ok, report.errors)
        self.assertEqual(
            observation.metadata["opponent_revealed"],
            1,
        )

    def test_vector_range_and_mask(self):
        battle = DummyBattle(4)
        observation = UniversalStateEncoder().encode(battle)
        report = ObservationValidator().validate(
            observation,
            battle=battle,
        )

        self.assertTrue(report.ok, report.errors)
        self.assertTrue(
            np.all(observation.vector >= -1.0)
        )
        self.assertTrue(
            np.all(observation.vector <= 1.0)
        )
        self.assertEqual(
            observation.legal_action_mask.shape,
            (10,),
        )


if __name__ == "__main__":
    unittest.main()
