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


    def test_transient_move_replaces_unavailable_known_slot(self):
        battle = DummyBattle(force_switch=False)
        space = UniversalActionSpace()

        known = [
            DummyMove("hyperbeam"),
            DummyMove("rest", base_power=0),
            DummyMove("earthquake"),
            DummyMove("bodyslam"),
        ]
        battle.active_pokemon.moves = {move.id: move for move in known}
        recharge = DummyMove("recharge", base_power=0)
        battle.available_moves = [recharge]

        slots = space.move_slots(battle)
        mask = space.legal_mask(battle)

        self.assertEqual(slots[0].id, "recharge")
        self.assertEqual(mask[:4].tolist(), [1, 0, 0, 0])
        self.assertEqual(space.index_for_selected(battle, recharge), 0)

    def test_transient_fight_is_mappable_with_four_known_moves(self):
        battle = DummyBattle(force_switch=False)
        space = UniversalActionSpace()

        known = [
            DummyMove("psychic"),
            DummyMove("recover", base_power=0),
            DummyMove("seismictoss"),
            DummyMove("thunderwave", base_power=0),
        ]
        battle.active_pokemon.moves = {move.id: move for move in known}
        fight = DummyMove("fight", base_power=0)
        battle.available_moves = [fight]

        slots = space.move_slots(battle)
        mask = space.legal_mask(battle)

        self.assertIn("fight", [move.id for move in slots if move is not None])
        index = space.index_for_selected(battle, fight)
        self.assertIsNotNone(index)
        self.assertEqual(mask[index], 1)

    def test_normal_known_move_order_is_preserved(self):
        battle = DummyBattle(force_switch=False)
        space = UniversalActionSpace()

        moves = [
            DummyMove("a"), DummyMove("b"),
            DummyMove("c"), DummyMove("d"),
        ]
        battle.active_pokemon.moves = {move.id: move for move in moves}
        battle.available_moves = [moves[0], moves[2]]

        slots = space.move_slots(battle)
        self.assertEqual([move.id for move in slots], ["a", "b", "c", "d"])


    def test_valid_orders_supplement_available_moves(self):
        battle = DummyBattle(force_switch=False)
        space = UniversalActionSpace()

        known = [
            DummyMove("stoneedge"),
            DummyMove("knockoff"),
            DummyMove("earthquake"),
            DummyMove("swordsdance", base_power=0),
        ]
        battle.active_pokemon.moves = {move.id: move for move in known}

        # Simulate the rare poke-env edge case observed during v0.5.5:
        # Earthquake is legal in valid_orders but absent from available_moves.
        battle.available_moves = [known[0], known[1]]

        class DummyOrder:
            def __init__(self, selected):
                self.order = selected
                self.mega = False
                self.z_move = False
                self.dynamax = False
                self.terastallize = False

        battle.valid_orders = [
            DummyOrder(known[0]),
            DummyOrder(known[1]),
            DummyOrder(known[2]),
        ]

        mask = space.legal_mask(battle)
        index = space.index_for_selected(battle, known[2])

        self.assertIsNotNone(index)
        self.assertEqual(mask[index], 1)


if __name__ == "__main__":
    unittest.main()
