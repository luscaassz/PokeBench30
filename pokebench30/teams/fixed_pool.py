from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "fixed_teams.yaml"
DEFAULT_POOL_ROOT = PROJECT_ROOT / "teams" / "fixed"


@dataclass(frozen=True)
class TeamRecord:
    generation: int
    pair_id: str
    side: str
    team_id: str
    seed: str
    source_format: str
    validation_format: str
    battle_format: str
    packed_path: str
    export_path: str
    sha256: str


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixed_config(path: Path = DEFAULT_CONFIG) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def config_hash(path: Path = DEFAULT_CONFIG) -> str:
    return sha256_file(Path(path))


def _run(
    command: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
) -> str:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\n\nSTDOUT:\n"
            + completed.stdout
            + "\nSTDERR:\n"
            + completed.stderr
        )
    return completed.stdout.strip()


def showdown_commit(showdown_dir: Path) -> str:
    try:
        return _run(
            ["git", "rev-parse", "HEAD"],
            cwd=showdown_dir,
        )
    except Exception:
        return "unknown"


def showdown_dirty(showdown_dir: Path) -> bool | None:
    try:
        output = _run(
            ["git", "status", "--porcelain"],
            cwd=showdown_dir,
        )
        return bool(output.strip())
    except Exception:
        return None


def node_version(showdown_dir: Path) -> str:
    try:
        return _run(["node", "--version"], cwd=showdown_dir)
    except Exception:
        return "unknown"


def _showdown_script(showdown_dir: Path) -> str:
    script = showdown_dir / "pokemon-showdown"
    if not script.exists():
        raise FileNotFoundError(
            f"Could not find {script}. "
            "Point --showdown-dir to the cloned pokemon-showdown repository."
        )
    return "pokemon-showdown"


def generate_packed_team(
    showdown_dir: Path,
    source_format: str,
    seed: str,
) -> str:
    script = _showdown_script(showdown_dir)
    packed = _run(
        ["node", script, "generate-team", source_format, seed],
        cwd=showdown_dir,
    )
    if not packed or "|" not in packed:
        raise RuntimeError(
            f"Unexpected team output for {source_format} seed {seed}: {packed!r}"
        )
    return packed.strip()


def validate_team(
    showdown_dir: Path,
    validation_format: str,
    packed_team: str,
) -> None:
    script = _showdown_script(showdown_dir)
    _run(
        ["node", script, "validate-team", validation_format],
        cwd=showdown_dir,
        input_text=packed_team + "\n",
    )


def export_team(showdown_dir: Path, packed_team: str) -> str:
    script = _showdown_script(showdown_dir)
    return _run(
        ["node", script, "export-team"],
        cwd=showdown_dir,
        input_text=packed_team + "\n",
    )


def prepare_fixed_pool(
    *,
    showdown_dir: Path,
    config_path: Path = DEFAULT_CONFIG,
    pool_root: Path = DEFAULT_POOL_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    showdown_dir = Path(showdown_dir).resolve()
    config_path = Path(config_path).resolve()
    pool_root = Path(pool_root).resolve()

    existing_manifest = pool_root / "manifest.json"
    if existing_manifest.exists() and not force:
        return json.loads(existing_manifest.read_text(encoding="utf-8"))

    if force and pool_root.exists():
        shutil.rmtree(pool_root)

    pool_root.mkdir(parents=True, exist_ok=True)
    config = load_fixed_config(config_path)

    records: list[TeamRecord] = []

    for generation_raw, gen_cfg in config["generations"].items():
        generation = int(generation_raw)
        gen_dir = pool_root / f"gen{generation}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        source_format = gen_cfg["source_format"]
        validation_format = gen_cfg["validation_format"]
        battle_format = gen_cfg["battle_format"]

        for pair in gen_cfg["pairs"]:
            pair_id = str(pair["id"])
            for side, seed_key in (("A", "team_a_seed"), ("B", "team_b_seed")):
                seed = str(pair[seed_key])
                team_id = f"gen{generation}_{pair_id}_{side}"

                packed = generate_packed_team(
                    showdown_dir,
                    source_format,
                    seed,
                )
                validate_team(
                    showdown_dir,
                    validation_format,
                    packed,
                )
                exported = export_team(showdown_dir, packed)

                packed_path = gen_dir / f"{pair_id}_{side}.packed.txt"
                export_path = gen_dir / f"{pair_id}_{side}.txt"

                packed_path.write_text(packed + "\n", encoding="utf-8")
                export_path.write_text(exported.rstrip() + "\n", encoding="utf-8")

                records.append(
                    TeamRecord(
                        generation=generation,
                        pair_id=pair_id,
                        side=side,
                        team_id=team_id,
                        seed=seed,
                        source_format=source_format,
                        validation_format=validation_format,
                        battle_format=battle_format,
                        packed_path=str(packed_path.relative_to(PROJECT_ROOT)),
                        export_path=str(export_path.relative_to(PROJECT_ROOT)),
                        sha256=sha256_text(packed),
                    )
                )

    manifest = {
        "schema_version": 1,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "config_sha256": config_hash(config_path),
        "showdown_dir": str(showdown_dir),
        "showdown_commit": showdown_commit(showdown_dir),
        "showdown_dirty": showdown_dirty(showdown_dir),
        "node_version": node_version(showdown_dir),
        "records": [asdict(record) for record in records],
    }

    existing_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_pool_manifest(
    pool_root: Path = DEFAULT_POOL_ROOT,
) -> dict[str, Any]:
    path = Path(pool_root) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Fixed-team manifest not found at {path}.\n"
            "Prepare the pool first with:\n"
            "  python prepare_fixed_teams.py --showdown-dir <pokemon-showdown>"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def records_by_key(
    manifest: dict[str, Any],
) -> dict[tuple[int, str, str], TeamRecord]:
    output = {}
    for raw in manifest["records"]:
        record = TeamRecord(**raw)
        output[(record.generation, record.pair_id, record.side)] = record
    return output


def read_packed_team(record: TeamRecord) -> str:
    path = PROJECT_ROOT / record.packed_path
    return path.read_text(encoding="utf-8").strip()


def verify_pool(
    pool_root: Path = DEFAULT_POOL_ROOT,
) -> list[str]:
    manifest = load_pool_manifest(pool_root)
    problems: list[str] = []

    for raw in manifest["records"]:
        record = TeamRecord(**raw)
        path = PROJECT_ROOT / record.packed_path

        if not path.exists():
            problems.append(f"Missing team file: {path}")
            continue

        actual = sha256_text(path.read_text(encoding="utf-8").strip())
        if actual != record.sha256:
            problems.append(
                f"Hash mismatch for {record.team_id}: "
                f"expected {record.sha256}, got {actual}"
            )

    return problems
