import unittest

from pokebench30.representation.action_space import (
    IllegalActionError,
    UniversalActionSpace,
)


class EnumLike:
    def __init__(self, name):
        self.name = name


class DummyMove:
    def __init__(self, move_id, base_power=80):
        self.id = move_id
        self.base_power = base_power
        self.accuracy = 1.0
        self.priority = 0
        self.type = EnumLike("NORMAL")
        self.category = EnumLike("PHYSICAL")


class DummyPokemon:
    def __init__(self, species, moves=None, active=False):
        self.species = species
        self.name = species
        self.level = 100
        self.moves = {
            move.id: move for move in (moves or [])
        }
        self.active = active
        self.fainted = False
        self.current_hp_fraction = 1.0


class DummyBattle:
    def __init__(self, force_switch=False):
        tackle = DummyMove("tackle")
        growl = DummyMove("growl", base_power=0)

        self.active_pokemon = DummyPokemon(
            "alpha",
            [tackle, growl],
            active=True,
        )
        self.bench = DummyPokemon("beta")
        self.team = {
            "p1a": self.active_pokemon,
            "p1b": self.bench,
        }
        self.opponent_team = {}
        self.opponent_active_pokemon = None
        self.available_moves = (
            [] if force_switch else [tackle]
        )
        self.available_switches = [self.bench]
        self.force_switch = force_switch


class UniversalActionSpaceTests(unittest.TestCase):
    def test_move_and_switch_mask(self):
        battle = DummyBattle(force_switch=False)
        space = UniversalActionSpace()
        mask = space.legal_mask(battle)

        self.assertEqual(mask.shape, (10,))
        self.assertEqual(mask[0], 1)
        self.assertEqual(mask[1], 0)
        self.assertEqual(mask[4], 0)  # active team slot
        self.assertEqual(mask[5], 1)  # bench team slot

    def test_forced_switch_masks_all_moves(self):
        battle = DummyBattle(force_switch=True)
        mask = UniversalActionSpace().legal_mask(battle)

        self.assertEqual(mask[:4].sum(), 0)
        self.assertEqual(mask[5], 1)

    def test_illegal_action_is_rejected(self):
        battle = DummyBattle(force_switch=False)
        space = UniversalActionSpace()

        with self.assertRaises(IllegalActionError):
            space.object_for_action(battle, 1)


if __name__ == "__main__":
    unittest.main()
