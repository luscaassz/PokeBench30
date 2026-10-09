from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from pokebench30.representation.encoder import UniversalStateEncoder
from pokebench30.representation.validator import ObservationValidator


class EnumLike:
    def __init__(self, name):
        self.name = name


class DummyMove:
    def __init__(self, move_id, type_name, category, power):
        self.id = move_id
        self.type = EnumLike(type_name)
        self.category = EnumLike(category)
        self.base_power = power
        self.accuracy = 1.0
        self.priority = 0


class DummyPokemon:
    def __init__(self, species, type_name, moves=None, active=False):
        self.species = species
        self.name = species
        self.type_1 = EnumLike(type_name)
        self.type_2 = None
        self.current_hp_fraction = 1.0
        self.fainted = False
        self.status = None
        self.level = 100
        self.boosts = {}
        self.active = active
        self.moves = {
            move.id: move for move in (moves or [])
        }
        self.effects = {}


class DummyBattle:
    def __init__(self, gen):
        move = DummyMove(
            "tackle",
            "NORMAL",
            "PHYSICAL",
            40,
        )

        self.gen = gen
        self.turn = 1
        self.battle_tag = f"offline-gen{gen}"
        self.force_switch = False

        self.active_pokemon = DummyPokemon(
            "self-active",
            "NORMAL",
            moves=[move],
            active=True,
        )
        self.bench = DummyPokemon(
            "self-bench",
            "WATER",
        )

        self.team = {
            "self-active": self.active_pokemon,
            "self-bench": self.bench,
        }

        self.opponent_active_pokemon = DummyPokemon(
            "opponent-revealed",
            "FIRE",
            active=True,
        )
        self.opponent_team = {
            "opponent-revealed": self.opponent_active_pokemon,
        }

        self.available_moves = [move]
        self.available_switches = [self.bench]

        self.side_conditions = {}
        self.opponent_side_conditions = {}
        self.weather = {}
        self.fields = {}

        # Exercise the UBR 1.2 context block in the Gen IX offline sample.
        if gen == 9:
            self.side_conditions = {
                EnumLike("STEALTH_ROCK"): 1,
                EnumLike("SPIKES"): 2,
            }
            self.opponent_side_conditions = {
                EnumLike("STICKY_WEB"): 1,
            }
            self.weather = {
                EnumLike("RAINDANCE"): 1,
            }
            self.fields = {
                EnumLike("ELECTRIC_TERRAIN"): 1,
            }
            self.active_pokemon.effects = {
                EnumLike("TAUNT"): 1,
            }


def main():
    parser = argparse.ArgumentParser(
        description="Offline check for a PokéBench30 universal representation profile."
    )
    parser.add_argument(
        "--ubr",
        choices=["ubr-1.1", "ubr-1.2"],
        default="ubr-1.2",
    )
    args = parser.parse_args()

    encoder = UniversalStateEncoder(representation_version=args.ubr)
    validator = ObservationValidator(args.ubr)

    print(
        f"[OK] UBR schema: {encoder.schema.version}"
    )
    print(
        f"[OK] Observation dimension: "
        f"{encoder.schema.observation_dim}"
    )
    print(
        f"[OK] Action dimension: "
        f"{encoder.schema.action_dim}"
    )
    print(
        f"[OK] Schema SHA-256: "
        f"{encoder.schema.sha256}"
    )

    vectors = []

    for generation in (1, 4, 9):
        battle = DummyBattle(generation)
        observation = encoder.encode(battle)
        report = validator.validate(
            observation,
            battle=battle,
        )

        if not report.ok:
            print(
                f"[FAIL] Gen {generation}: "
                + "; ".join(report.errors)
            )
            raise SystemExit(1)

        vectors.append(observation.vector)

        print(
            f"[OK] Gen {generation}: "
            f"shape={observation.vector.shape}, "
            f"legal_actions="
            f"{int(observation.legal_action_mask.sum())}, "
            f"opponent_revealed="
            f"{observation.metadata['opponent_revealed']}, "
            f"context_nonzero="
            f"{sum(observation.metadata.get('battle_context', {}).values())}"
        )

    if len({vector.shape for vector in vectors}) != 1:
        print("[FAIL] Generations produced different shapes")
        raise SystemExit(1)

    if any(
        not np.all(np.isfinite(vector))
        for vector in vectors
    ):
        print("[FAIL] Non-finite value found")
        raise SystemExit(1)

    print(
        "\n[OK] Universal representation offline checks passed."
    )


if __name__ == "__main__":
    main()
