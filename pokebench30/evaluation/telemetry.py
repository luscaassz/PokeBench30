from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


MOVE_FEATURES = (
    "damaging",
    "status",
    "self_boost",
    "self_debuff",
    "target_boost",
    "target_debuff",
    "any_buff",
    "any_debuff",
    "status_inflict",
    "volatile_status",
    "healing",
    "drain",
    "recoil",
    "hazard_set",
    "hazard_remove",
    "weather_set",
    "terrain_set",
    "priority",
    "protect_like",
    "pivot_like",
    "force_switch",
)

PROTECT_MOVE_IDS = {
    "protect", "detect", "kingsshield", "spikyshield", "banefulbunker",
    "obstruct", "silktrap", "burningbulwark", "maxguard",
}
PIVOT_MOVE_IDS = {
    "uturn", "voltswitch", "flipturn", "partingshot", "batonpass",
    "teleport", "chillyreception",
}
HAZARD_MOVE_IDS = {
    "stealthrock", "spikes", "toxicspikes", "stickyweb", "stoneaxe",
    "ceaselessedge",
}
HAZARD_REMOVAL_MOVE_IDS = {
    "rapidspin", "defog", "mortalspin", "tidyup",
}

# National Pokédex upper bounds for the base-species structural catalog.
# v0.6.1-hotfix1 deliberately restricts the broad generation catalog to
# canonical base species only. Form-level ecology remains available from the
# frozen benchmark pool, which is the correct place to study forms actually
# encountered by the benchmark.
NATIONAL_DEX_MAX_BY_GENERATION = {
    1: 151,
    2: 251,
    3: 386,
    4: 493,
    5: 649,
    6: 721,
    7: 809,
    8: 905,
    9: 1025,
}

SELF_TARGET_NAMES = {
    "SELF",
    "ALLYTEAM",
}


def _id_like(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _is_base_species_catalog_entry(
    species_id: str,
    raw: dict[str, Any],
    generation: int,
) -> bool:
    """Return whether a Pokédex row belongs in the broad base-species catalog.

    poke-env's generation Pokédex files can contain aliases, cosmetic forms,
    later-generation forms, CAP entries and other rows that are useful for
    battle parsing but are not valid evidence that those species/forms existed
    in the historical generation. For the generation-wide structural layer we
    therefore count only canonical base species with a positive National Dex
    number introduced by that generation.

    Form-level composition is measured separately from the frozen fixed pool.
    """
    generation = int(generation)
    max_num = NATIONAL_DEX_MAX_BY_GENERATION.get(generation)
    if max_num is None:
        return False

    try:
        num = int(raw.get("num", 0) or 0)
    except (TypeError, ValueError):
        return False
    if num <= 0 or num > max_num:
        return False

    # Forms (Mega, regional forms, Gmax, Rotom forms, etc.) are intentionally
    # excluded from this broad catalog. They remain visible in the fixed-pool
    # catalog when actually used by the benchmark.
    if str(raw.get("forme", "") or "").strip():
        return False

    # Cosmetic aliases inserted by GenData.load_pokedex point to the same raw
    # object. Keep only the canonical id corresponding to the row's own name.
    canonical_id = _id_like(raw.get("name", species_id))
    if canonical_id and canonical_id != _id_like(species_id):
        return False

    base_species = _id_like(raw.get("baseSpecies", raw.get("name", species_id)))
    if base_species and base_species != _id_like(species_id):
        return False

    return True


def _canonical_manifest_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Mirror fixed_pool.records_by_key semantics for telemetry catalogs.

    Some legacy manifests contain repeated records for the same
    generation/pair/side. The battle evaluator resolves those records by key
    with the last entry winning. Static telemetry must use the exact same rule
    or the old pairs receive double weight.
    """
    records: dict[tuple[int, str, str], dict[str, Any]] = {}
    for raw in manifest.get("records", []):
        key = (
            int(raw["generation"]),
            str(raw["pair_id"]),
            str(raw["side"]),
        )
        records[key] = raw
    return [records[key] for key in sorted(records)]


def safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except (AttributeError, KeyError, TypeError, ValueError):
        return default


def enum_name(value: Any) -> str:
    if value is None:
        return ""
    name = safe_attr(value, "name", None)
    return str(name if name is not None else value)


def _move_id(move: Any) -> str:
    if move is None:
        return ""
    return str(
        safe_attr(move, "id", None)
        or safe_attr(move, "name", None)
        or move
    ).lower().replace(" ", "")


def _numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _mapping(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if hasattr(value, "items"):
        return {str(enum_name(k)).lower(): v for k, v in value.items()}
    return {}


def _boost_mapping(value: Any) -> dict[str, Any]:
    mapping = _mapping(value)
    nested = mapping.get("boosts")
    if isinstance(nested, dict):
        return {str(key).lower(): val for key, val in nested.items()}
    return mapping


def _has_positive(mapping: dict[str, Any]) -> bool:
    for value in mapping.values():
        number = _numeric(value)
        if number is not None and number > 0:
            return True
    return False


def _has_negative(mapping: dict[str, Any]) -> bool:
    for value in mapping.values():
        number = _numeric(value)
        if number is not None and number < 0:
            return True
    return False


def move_semantics(move: Any) -> dict[str, Any]:
    """Return version-tolerant, multi-label move telemetry.

    The tags deliberately remain multi-label. A damaging move can also inflict
    status, debuff a target, drain HP, pivot, etc.
    """
    move_id = _move_id(move)
    category = enum_name(safe_attr(move, "category", None)).upper()
    move_type = enum_name(safe_attr(move, "type", None)).upper()

    raw_boosts = _boost_mapping(safe_attr(move, "boosts", {}))
    explicit_self_boosts = _boost_mapping(
        safe_attr(move, "self_boost", None)
        or safe_attr(move, "self_boosts", None)
        or {}
    )

    base_power = _numeric(safe_attr(move, "base_power", None))
    accuracy = _numeric(safe_attr(move, "accuracy", None))
    priority = _numeric(safe_attr(move, "priority", 0.0)) or 0.0
    heal = _numeric(safe_attr(move, "heal", None))
    drain = _numeric(safe_attr(move, "drain", None))
    recoil = _numeric(safe_attr(move, "recoil", None))

    side_condition = enum_name(safe_attr(move, "side_condition", None)).upper()
    weather = enum_name(safe_attr(move, "weather", None)).upper()
    terrain = enum_name(
        safe_attr(move, "terrain", None)
        or safe_attr(move, "field", None)
    ).upper()
    status = enum_name(safe_attr(move, "status", None)).upper()
    volatile = enum_name(safe_attr(move, "volatile_status", None)).upper()
    target = enum_name(safe_attr(move, "target", None)).upper()

    # In Pokémon Showdown metadata, `boosts` describes the main target of the
    # move. For SELF-targeting moves such as Swords Dance or Agility, those
    # boosts belong to the user, not the opponent. Earlier v0.6.1 telemetry
    # incorrectly counted them as target boosts/debuffs. Route the main boost
    # map according to the target and merge any explicit `self_boost` metadata.
    main_targets_self = target in SELF_TARGET_NAMES
    target_boosts = {} if main_targets_self else dict(raw_boosts)
    self_boosts = dict(explicit_self_boosts)
    if main_targets_self:
        for stat, value in raw_boosts.items():
            # Explicit self metadata wins if both sources describe the same stat.
            self_boosts.setdefault(stat, value)

    self_switch_raw = safe_attr(move, "self_switch", None)
    force_switch_raw = safe_attr(move, "force_switch", None)

    is_status = category == "STATUS"
    is_damaging = category in {"PHYSICAL", "SPECIAL"}
    # A conservative fallback for incomplete metadata.
    if not category and base_power is not None and base_power > 0:
        is_damaging = True

    hazard_set = bool(side_condition) or move_id in HAZARD_MOVE_IDS
    hazard_remove = move_id in HAZARD_REMOVAL_MOVE_IDS
    pivot_like = bool(self_switch_raw) or move_id in PIVOT_MOVE_IDS
    protect_like = move_id in PROTECT_MOVE_IDS

    return {
        "move_id": move_id,
        "move_name": str(safe_attr(move, "name", move_id) or move_id),
        "move_type": move_type,
        "move_category": category,
        "base_power": base_power,
        "accuracy": accuracy,
        "priority_value": priority,
        "target": target,
        "status_name": status,
        "volatile_status_name": volatile,
        "side_condition": side_condition,
        "weather": weather,
        "terrain": terrain,
        "boosts_json": json.dumps(target_boosts, sort_keys=True, default=str),
        "self_boosts_json": json.dumps(self_boosts, sort_keys=True, default=str),
        "feature_damaging": bool(is_damaging),
        "feature_status": bool(is_status),
        "feature_self_boost": bool(_has_positive(self_boosts)),
        "feature_self_debuff": bool(_has_negative(self_boosts)),
        "feature_target_boost": bool(_has_positive(target_boosts)),
        "feature_target_debuff": bool(_has_negative(target_boosts)),
        "feature_any_buff": bool(_has_positive(self_boosts) or _has_positive(target_boosts)),
        "feature_any_debuff": bool(_has_negative(self_boosts) or _has_negative(target_boosts)),
        "feature_status_inflict": bool(status),
        "feature_volatile_status": bool(volatile),
        "feature_healing": bool(heal is not None and heal > 0),
        "feature_drain": bool(drain is not None and drain > 0),
        "feature_recoil": bool(recoil is not None and recoil != 0),
        "feature_hazard_set": bool(hazard_set),
        "feature_hazard_remove": bool(hazard_remove),
        "feature_weather_set": bool(weather),
        "feature_terrain_set": bool(terrain),
        "feature_priority": bool(priority != 0),
        "feature_protect_like": bool(protect_like),
        "feature_pivot_like": bool(pivot_like),
        "feature_force_switch": bool(force_switch_raw),
    }


def legal_move_profiles(battle: Any) -> list[dict[str, Any]]:
    # Import lazily to avoid importing representation modules in offline
    # telemetry-only analysis.
    from pokebench30.representation.action_space import _legal_regular_moves

    return [move_semantics(move) for move in _legal_regular_moves(battle)]


def selected_move_telemetry(move: Any) -> dict[str, Any]:
    sem = move_semantics(move)
    row = {
        "move_priority": sem["priority_value"],
        "move_target": sem["target"],
        "move_status_name": sem["status_name"],
        "move_volatile_status_name": sem["volatile_status_name"],
        "move_side_condition": sem["side_condition"],
        "move_weather": sem["weather"],
        "move_terrain": sem["terrain"],
        "move_boosts": sem["boosts_json"],
        "move_self_boosts": sem["self_boosts_json"],
    }
    for feature in MOVE_FEATURES:
        row[f"move_feature_{feature}"] = sem[f"feature_{feature}"]
    return row


def opportunity_telemetry(battle: Any) -> dict[str, Any]:
    profiles = legal_move_profiles(battle)
    out: dict[str, Any] = {
        "available_move_profiles_json": json.dumps(
            profiles, ensure_ascii=False, separators=(",", ":")
        ),
        "available_move_ids_json": json.dumps(
            [profile["move_id"] for profile in profiles],
            ensure_ascii=False,
        ),
        "available_move_types_json": json.dumps(
            sorted({profile["move_type"] for profile in profiles if profile["move_type"]}),
            ensure_ascii=False,
        ),
        "available_move_categories_json": json.dumps(
            sorted({profile["move_category"] for profile in profiles if profile["move_category"]}),
            ensure_ascii=False,
        ),
    }
    for feature in MOVE_FEATURES:
        out[f"available_count_{feature}"] = sum(
            bool(profile[f"feature_{feature}"]) for profile in profiles
        )
    return out


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float | None:
    return _numeric(value)


def _json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    try:
        decoded = json.loads(str(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    return decoded if isinstance(decoded, list) else []


def _rl_rows(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in actions if row.get("agent_role") == "rl"]
    # Backward-compatible fallback for standalone telemetry tests/logs.
    if not rows:
        rows = [row for row in actions if row.get("agent") == "RLPolicyAgent"]
    return rows


def behavior_summary(actions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _rl_rows(actions)
    decisions = len(rows)
    moves = [row for row in rows if row.get("action_type") == "move"]
    switches = [row for row in rows if row.get("action_type") == "switch"]

    def rate(count: int, denom: int = decisions) -> float:
        return float(count / denom) if denom else 0.0

    selected_bp = [
        value for row in moves
        if (value := _float(row.get("move_base_power"))) is not None
    ]
    selected_acc = [
        value for row in moves
        if (value := _float(row.get("move_accuracy"))) is not None
    ]
    selected_eff = [
        value for row in moves
        if (value := _float(row.get("move_type_effectiveness"))) is not None
    ]

    out: dict[str, Any] = {
        "decisions": decisions,
        "move_actions": len(moves),
        "switch_actions": len(switches),
        "move_rate": rate(len(moves)),
        "switch_rate": rate(len(switches)),
        "voluntary_switch_rate": rate(sum(_truthy(r.get("voluntary_switch")) for r in rows)),
        "forced_switch_rate": rate(sum(_truthy(r.get("forced_switch")) for r in rows)),
        "mean_selected_base_power": fmean(selected_bp) if selected_bp else "",
        "mean_selected_accuracy": fmean(selected_acc) if selected_acc else "",
        "stab_move_rate": rate(sum((_float(r.get("move_stab")) or 1.0) > 1.0 for r in moves), len(moves)),
        "super_effective_rate": rate(sum(v > 1.0 for v in selected_eff), len(selected_eff)),
        "neutral_effective_rate": rate(sum(abs(v - 1.0) < 1e-9 for v in selected_eff), len(selected_eff)),
        "resisted_rate": rate(sum(v < 1.0 for v in selected_eff), len(selected_eff)),
    }

    for feature in MOVE_FEATURES:
        selected = sum(
            _truthy(row.get(f"move_feature_{feature}"))
            for row in moves
        )
        available_decisions = sum(
            int(_float(row.get(f"available_count_{feature}")) or 0) > 0
            for row in rows
        )
        selected_when_available = sum(
            row.get("action_type") == "move"
            and _truthy(row.get(f"move_feature_{feature}"))
            and int(_float(row.get(f"available_count_{feature}")) or 0) > 0
            for row in rows
        )
        out[f"selected_{feature}"] = selected
        out[f"selected_{feature}_rate"] = rate(selected, len(moves))
        out[f"availability_decisions_{feature}"] = available_decisions
        out[f"availability_rate_{feature}"] = rate(available_decisions)
        out[f"selection_given_available_{feature}"] = rate(
            selected_when_available,
            available_decisions,
        )

    return out


def explode_opportunities(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in _rl_rows(actions):
        profiles = _json_list(row.get("available_move_profiles_json"))
        selected_id = str(row.get("action", "")).lower().replace(" ", "")
        base = {
            key: row.get(key, "")
            for key in (
                "battle_tag", "generation", "battle_format", "pair_id",
                "mirror_leg", "repeat_index", "request_index", "turn",
                "agent_team_id", "opponent_team_id", "active_pokemon",
                "opponent_pokemon",
            )
        }
        for profile in profiles:
            item = dict(base)
            item.update(profile)
            item["selected"] = (
                row.get("action_type") == "move"
                and str(profile.get("move_id", "")) == selected_id
            )
            output.append(item)
    return output


def _usage_table(
    actions: list[dict[str, Any]],
    *,
    key: str,
) -> list[dict[str, Any]]:
    rows = _rl_rows(actions)
    selected = Counter()
    available_decisions = Counter()

    for row in rows:
        if row.get("action_type") == "move":
            if key == "move_id":
                value = str(row.get("action", "")).lower().replace(" ", "")
            elif key == "move_type":
                value = str(row.get("move_type", ""))
            else:
                value = str(row.get("move_category", ""))
            if value:
                selected[value] += 1

        profiles = _json_list(row.get("available_move_profiles_json"))
        present: set[str] = set()
        for profile in profiles:
            value = str(profile.get(key, ""))
            if value:
                present.add(value)
        for value in present:
            available_decisions[value] += 1

    values = sorted(set(selected) | set(available_decisions))
    output = []
    for value in values:
        avail = available_decisions[value]
        sel = selected[value]
        output.append({
            key: value,
            "selected_count": sel,
            "available_decisions": avail,
            "selection_given_available": (sel / avail) if avail else 0.0,
            "selected_share": (sel / sum(selected.values())) if selected else 0.0,
        })
    return output


def move_usage(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _usage_table(actions, key="move_id")


def move_type_usage(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _usage_table(actions, key="move_type")


def move_category_usage(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _usage_table(actions, key="move_category")


def move_feature_usage(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = behavior_summary(actions)
    output = []
    for feature in MOVE_FEATURES:
        output.append({
            "feature": feature,
            "selected_count": summary[f"selected_{feature}"],
            "selected_rate_among_moves": summary[f"selected_{feature}_rate"],
            "availability_decisions": summary[f"availability_decisions_{feature}"],
            "availability_rate": summary[f"availability_rate_{feature}"],
            "selection_given_available": summary[f"selection_given_available_{feature}"],
        })
    return output


def pokemon_type_exposure(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _rl_rows(actions)
    counts: dict[str, Counter] = {
        "active": Counter(),
        "opponent_active": Counter(),
        "switch_target": Counter(),
    }
    for row in rows:
        for field in ("active_type_1", "active_type_2"):
            value = str(row.get(field, ""))
            if value:
                counts["active"][value] += 1
        for field in ("opponent_type_1", "opponent_type_2"):
            value = str(row.get(field, ""))
            if value:
                counts["opponent_active"][value] += 1
        if row.get("action_type") == "switch":
            for field in ("switch_target_type_1", "switch_target_type_2"):
                value = str(row.get(field, ""))
                if value:
                    counts["switch_target"][value] += 1

    output = []
    for context, counter in counts.items():
        total = sum(counter.values())
        for type_name, count in sorted(counter.items()):
            output.append({
                "context": context,
                "pokemon_type": type_name,
                "count": count,
                "share": count / total if total else 0.0,
            })
    return output


def generation_dex_catalog(
    generations: Iterable[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Catalog generation data shipped with poke-env.

    This is the broadest structural layer in the telemetry stack: it describes
    the move and Pokédex entries available in poke-env's per-generation data,
    independently of the frozen benchmark teams or any learned policy.
    """
    try:
        from poke_env.battle.move import Move
        from poke_env.data import GenData
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Generation dex telemetry requires poke-env.") from exc

    pokemon_rows: list[dict[str, Any]] = []
    move_rows: list[dict[str, Any]] = []

    for generation_raw in generations:
        generation = int(generation_raw)
        data = GenData.from_gen(generation)

        for species_id, raw in sorted(data.pokedex.items()):
            if not _is_base_species_catalog_entry(species_id, raw, generation):
                continue
            types = list(raw.get("types", []) or [])
            pokemon_rows.append({
                "generation": generation,
                "species_id": species_id,
                "name": raw.get("name", species_id),
                "base_species": raw.get("baseSpecies", raw.get("name", species_id)),
                "forme": raw.get("forme", ""),
                "national_dex_number": int(raw.get("num", 0) or 0),
                "type_1": str(types[0]).upper() if types else "",
                "type_2": str(types[1]).upper() if len(types) > 1 else "",
                "is_nonstandard": str(raw.get("isNonstandard", "") or ""),
                "catalog_scope": "canonical_base_species_introduced_by_generation",
            })

        for move_id, raw in sorted(data.moves.items()):
            try:
                sem = move_semantics(Move(move_id, gen=generation))
            except Exception as exc:
                sem = {
                    "move_id": move_id,
                    "move_name": raw.get("name", move_id),
                    "move_type": str(raw.get("type", "") or "").upper(),
                    "move_category": str(raw.get("category", "") or "").upper(),
                    "base_power": raw.get("basePower", ""),
                    "accuracy": raw.get("accuracy", ""),
                    "priority_value": raw.get("priority", ""),
                    "target": str(raw.get("target", "") or "").upper(),
                    "status_name": "",
                    "volatile_status_name": "",
                    "side_condition": "",
                    "weather": "",
                    "terrain": "",
                    "boosts_json": "{}",
                    "self_boosts_json": "{}",
                    **{f"feature_{feature}": False for feature in MOVE_FEATURES},
                    "catalog_error": str(exc),
                }
            move_rows.append({
                "generation": generation,
                **sem,
                "is_nonstandard": str(raw.get("isNonstandard", "") or ""),
            })

    return pokemon_rows, move_rows


def parse_packed_team(packed: str) -> list[dict[str, Any]]:
    """Parse the stable core of Pokémon Showdown's packed team format.

    Only fields needed by benchmark ecology telemetry are extracted. Unknown or
    generation-specific trailing fields are intentionally preserved neither as
    assumptions nor fabricated values.
    """
    output: list[dict[str, Any]] = []
    for slot, raw_set in enumerate(str(packed).strip().split("]"), start=1):
        if not raw_set:
            continue
        fields = raw_set.split("|")
        while len(fields) < 5:
            fields.append("")
        nickname = fields[0]
        species = fields[1] or nickname
        item = fields[2] if len(fields) > 2 else ""
        ability = fields[3] if len(fields) > 3 else ""
        moves = [value for value in fields[4].split(",") if value]
        level = fields[11] if len(fields) > 11 else ""
        output.append({
            "team_slot": slot,
            "nickname": nickname,
            "species_raw": species,
            "item": item,
            "ability": ability,
            "move_ids": moves,
            "level": level,
        })
    return output


def fixed_pool_catalog_from_manifest(
    manifest: dict[str, Any],
    *,
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a static catalog directly from frozen packed teams.

    This is preferred over post-battle Pokémon objects for generation-ecology
    analysis because battle transformations, Mimic/Transform and temporary
    form changes cannot alter the source catalog.
    """
    try:
        from poke_env.battle.move import Move
        from poke_env.data import GenData, to_id_str
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Static fixed-pool telemetry requires poke-env."
        ) from exc

    pokemon_rows: list[dict[str, Any]] = []
    move_rows: list[dict[str, Any]] = []
    gen_cache: dict[int, Any] = {}

    for record in _canonical_manifest_records(manifest):
        generation = int(record["generation"])
        gen_data = gen_cache.setdefault(generation, GenData.from_gen(generation))
        packed_path = Path(project_root) / str(record["packed_path"])
        packed = packed_path.read_text(encoding="utf-8").strip()

        for parsed in parse_packed_team(packed):
            species_id = to_id_str(parsed["species_raw"])
            dex = gen_data.pokedex.get(species_id, {})
            types = list(dex.get("types", []) or [])
            pokemon_row = {
                "generation": generation,
                "pair_id": str(record["pair_id"]),
                "side": str(record["side"]),
                "team_id": str(record["team_id"]),
                "team_slot": int(parsed["team_slot"]),
                "species": species_id,
                "type_1": str(types[0]).upper() if types else "",
                "type_2": str(types[1]).upper() if len(types) > 1 else "",
                "level": parsed["level"],
                "ability": parsed["ability"],
                "item": parsed["item"],
                "catalog_source": "packed_fixed_team",
            }
            pokemon_rows.append(pokemon_row)

            for move_slot, move_id in enumerate(parsed["move_ids"], start=1):
                move_row = dict(pokemon_row)
                move_row["move_slot"] = move_slot
                try:
                    move = Move(str(move_id), gen=generation)
                    move_row.update(move_semantics(move))
                except Exception as exc:
                    move_row.update({
                        "move_id": str(move_id),
                        "move_name": str(move_id),
                        "move_type": "",
                        "move_category": "",
                        "base_power": "",
                        "accuracy": "",
                        "priority_value": "",
                        "target": "",
                        "status_name": "",
                        "volatile_status_name": "",
                        "side_condition": "",
                        "weather": "",
                        "terrain": "",
                        "boosts_json": "{}",
                        "self_boosts_json": "{}",
                        **{f"feature_{feature}": False for feature in MOVE_FEATURES},
                        "catalog_error": str(exc),
                    })
                move_rows.append(move_row)

    return pokemon_rows, move_rows


def team_catalog(
    battle: Any,
    *,
    generation: int,
    team_id: str,
    pair_id: str,
    side: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    team = safe_attr(battle, "team", {}) or {}
    pokemon = list(team.values()) if hasattr(team, "values") else list(team)
    pokemon_rows: list[dict[str, Any]] = []
    move_rows: list[dict[str, Any]] = []

    for slot, mon in enumerate(pokemon, start=1):
        type_1 = enum_name(safe_attr(mon, "type_1", None)).upper()
        type_2 = enum_name(safe_attr(mon, "type_2", None)).upper()
        species = str(safe_attr(mon, "species", "") or "")
        mon_row = {
            "generation": int(generation),
            "pair_id": pair_id,
            "side": side,
            "team_id": team_id,
            "team_slot": slot,
            "species": species,
            "type_1": type_1,
            "type_2": type_2,
            "level": safe_attr(mon, "level", ""),
            "ability": str(safe_attr(mon, "ability", "") or ""),
            "item": str(safe_attr(mon, "item", "") or ""),
            "catalog_source": "battle_observation",
        }
        pokemon_rows.append(mon_row)

        moves = safe_attr(mon, "moves", {}) or {}
        values = list(moves.values()) if hasattr(moves, "values") else list(moves)
        for move_slot, move in enumerate(values, start=1):
            move_row = dict(mon_row)
            move_row["move_slot"] = move_slot
            move_row.update(move_semantics(move))
            move_rows.append(move_row)

    return pokemon_rows, move_rows


def dedupe_dict_rows(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    output: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        output[tuple(row.get(key) for key in keys)] = row
    return list(output.values())


FIXED_POOL_STATIC_CATALOG_SOURCE = "packed_fixed_team"


def canonical_fixed_pool_catalog_rows(
    pokemon_rows: Iterable[dict[str, Any]],
    move_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return one immutable catalog row per frozen team slot / move slot.

    The structural fixed-pool catalog must be built from packed team files,
    never from post-battle team observations. Battle-side telemetry may contain
    transformed state (e.g. Ditto) and, across mirrored/repeated runs, extra
    slot observations. Mixing both sources caused hotfix1 aggregates to grow
    from 120 to 156 Pokémon rows per generation.

    Normal v0.6.1 transfer studies always include ``catalog_source`` equal to
    ``packed_fixed_team`` for the immutable rows. A fallback to all rows is
    retained only for older tests/ad-hoc callers that do not provide source
    metadata.
    """
    pokemon_all = list(pokemon_rows)
    move_all = list(move_rows)

    pokemon_static = [
        row for row in pokemon_all
        if str(row.get("catalog_source", "")) == FIXED_POOL_STATIC_CATALOG_SOURCE
    ]
    move_static = [
        row for row in move_all
        if str(row.get("catalog_source", "")) == FIXED_POOL_STATIC_CATALOG_SOURCE
    ]

    pokemon_source = pokemon_static if pokemon_static else pokemon_all
    move_source = move_static if move_static else move_all

    pokemon = dedupe_dict_rows(
        pokemon_source,
        ("generation", "pair_id", "side", "team_slot"),
    )
    moves = dedupe_dict_rows(
        move_source,
        ("generation", "pair_id", "side", "team_slot", "move_slot"),
    )

    pokemon.sort(key=lambda row: (
        int(row.get("generation", 0) or 0),
        str(row.get("pair_id", "")),
        str(row.get("side", "")),
        int(row.get("team_slot", 0) or 0),
    ))
    moves.sort(key=lambda row: (
        int(row.get("generation", 0) or 0),
        str(row.get("pair_id", "")),
        str(row.get("side", "")),
        int(row.get("team_slot", 0) or 0),
        int(row.get("move_slot", 0) or 0),
    ))
    return pokemon, moves


def validate_fixed_pool_catalog_rows(
    pokemon_rows: Iterable[dict[str, Any]],
    move_rows: Iterable[dict[str, Any]],
    *,
    generations: Iterable[int] = (1, 4, 9),
    expected_pairs: int = 10,
    expected_team_size: int = 6,
) -> None:
    """Fail fast if structural fixed-pool telemetry is duplicated/incomplete."""
    pokemon = list(pokemon_rows)
    moves = list(move_rows)

    pokemon_keys = [
        (
            int(row.get("generation", 0) or 0),
            str(row.get("pair_id", "")),
            str(row.get("side", "")),
            int(row.get("team_slot", 0) or 0),
        )
        for row in pokemon
    ]
    if len(pokemon_keys) != len(set(pokemon_keys)):
        raise ValueError("Duplicate fixed-pool Pokémon keys remain after canonicalization.")

    move_keys = [
        (
            int(row.get("generation", 0) or 0),
            str(row.get("pair_id", "")),
            str(row.get("side", "")),
            int(row.get("team_slot", 0) or 0),
            int(row.get("move_slot", 0) or 0),
        )
        for row in moves
    ]
    if len(move_keys) != len(set(move_keys)):
        raise ValueError("Duplicate fixed-pool move-slot keys remain after canonicalization.")

    for generation_raw in generations:
        generation = int(generation_raw)
        gen_pokemon = [
            row for row in pokemon
            if int(row.get("generation", 0) or 0) == generation
        ]
        expected_pokemon = int(expected_pairs) * 2 * int(expected_team_size)
        if len(gen_pokemon) != expected_pokemon:
            raise ValueError(
                f"Gen {generation} fixed-pool Pokémon catalog must contain "
                f"{expected_pokemon} rows; found {len(gen_pokemon)}."
            )

        team_counts: dict[tuple[str, str], int] = defaultdict(int)
        for row in gen_pokemon:
            team_counts[(str(row.get("pair_id", "")), str(row.get("side", "")))] += 1

        expected_teams = int(expected_pairs) * 2
        if len(team_counts) != expected_teams:
            raise ValueError(
                f"Gen {generation} fixed-pool catalog must contain "
                f"{expected_teams} pair/side teams; found {len(team_counts)}."
            )

        bad_teams = {
            key: count for key, count in team_counts.items()
            if count != int(expected_team_size)
        }
        if bad_teams:
            raise ValueError(
                f"Gen {generation} fixed-pool team sizes are invalid: {bad_teams}"
            )

        gen_moves = [
            row for row in moves
            if int(row.get("generation", 0) or 0) == generation
        ]
        moves_by_mon: dict[tuple[str, str, int], list[int]] = defaultdict(list)
        for row in gen_moves:
            moves_by_mon[
                (
                    str(row.get("pair_id", "")),
                    str(row.get("side", "")),
                    int(row.get("team_slot", 0) or 0),
                )
            ].append(int(row.get("move_slot", 0) or 0))

        for mon_key, slots in moves_by_mon.items():
            if len(slots) > 4 or any(slot < 1 or slot > 4 for slot in slots):
                raise ValueError(
                    f"Gen {generation} invalid move slots for {mon_key}: {slots}"
                )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def write_gzip_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def battle_dynamics_summary(battles: list[dict[str, Any]]) -> dict[str, Any]:
    def numbers(field: str) -> list[float]:
        output = []
        for row in battles:
            value = _float(row.get(field))
            if value is not None:
                output.append(value)
        return output

    turns = numbers("turns")
    requests = numbers("agent_1_action_requests")
    moves = numbers("agent_1_moves")
    switches = numbers("agent_1_switches")
    voluntary = numbers("agent_1_voluntary_switches")
    remaining_rl = numbers("agent_1_remaining_true")
    remaining_opp = numbers("agent_2_remaining_true")
    reveal = numbers("agent_1_opponent_reveal_fraction")

    n = len(battles)
    wins = sum(row.get("result_agent_1") == "win" for row in battles)
    losses = sum(row.get("result_agent_1") == "loss" for row in battles)
    draws = n - wins - losses

    def mean(values: list[float]) -> float | str:
        return fmean(values) if values else ""

    return {
        "battles": n,
        "wins_rl": wins,
        "losses_rl": losses,
        "draws": draws,
        "raw_win_rate_rl": wins / n if n else 0.0,
        "mean_turns": mean(turns),
        "mean_action_requests_rl": mean(requests),
        "mean_moves_rl": mean(moves),
        "mean_switches_rl": mean(switches),
        "mean_voluntary_switches_rl": mean(voluntary),
        "mean_remaining_rl": mean(remaining_rl),
        "mean_remaining_opponent": mean(remaining_opp),
        "mean_opponent_reveal_fraction": mean(reveal),
    }


def write_telemetry_outputs(
    run_dir: Path,
    *,
    actions: list[dict[str, Any]],
    battles: list[dict[str, Any]],
    team_pokemon_rows: list[dict[str, Any]],
    team_move_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    behavior = behavior_summary(actions)
    battle_summary = battle_dynamics_summary(battles)
    move_rows = move_usage(actions)
    type_rows = move_type_usage(actions)
    category_rows = move_category_usage(actions)
    feature_rows = move_feature_usage(actions)
    pokemon_types = pokemon_type_exposure(actions)
    opportunities = explode_opportunities(actions)

    team_pokemon_rows = dedupe_dict_rows(
        team_pokemon_rows,
        ("generation", "team_id", "team_slot", "species"),
    )
    team_move_rows = dedupe_dict_rows(
        team_move_rows,
        ("generation", "team_id", "team_slot", "species", "move_id"),
    )

    write_csv(run_dir / "telemetry_behavior_summary.csv", [behavior])
    write_csv(run_dir / "telemetry_battle_dynamics.csv", [battle_summary])
    write_csv(run_dir / "telemetry_move_usage.csv", move_rows)
    write_csv(run_dir / "telemetry_move_type_usage.csv", type_rows)
    write_csv(run_dir / "telemetry_move_category_usage.csv", category_rows)
    write_csv(run_dir / "telemetry_move_feature_usage.csv", feature_rows)
    write_csv(run_dir / "telemetry_pokemon_type_exposure.csv", pokemon_types)
    write_csv(run_dir / "telemetry_team_pokemon.csv", team_pokemon_rows)
    write_csv(run_dir / "telemetry_team_moves.csv", team_move_rows)
    write_gzip_csv(run_dir / "telemetry_move_opportunities.csv.gz", opportunities)

    manifest = {
        "schema_version": 1,
        "scope": "controlled_evaluation",
        "action_rows_total": len(actions),
        "battle_rows_total": len(battles),
        "rl_decision_rows": len(_rl_rows(actions)),
        "opportunity_rows": len(opportunities),
        "team_pokemon_rows": len(team_pokemon_rows),
        "team_move_rows": len(team_move_rows),
        "move_features": list(MOVE_FEATURES),
        "notes": [
            "Move semantics are multi-label.",
            "Selection-given-available uses legal actions visible at the decision point.",
            "Fixed-pool team catalog describes the benchmark pool, not the full generation metagame.",
            "Detailed opportunity rows include RL decisions only; opponent actions remain in actions.csv.",
        ],
    }
    (run_dir / "telemetry_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
