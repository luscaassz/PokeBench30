from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_agent_summary(run_dir: Path) -> list[dict]:
    battles = _read_csv(run_dir / "battles.csv")
    actions_path = run_dir / "actions.csv"
    actions = _read_csv(actions_path) if actions_path.exists() else []

    stats = defaultdict(lambda: {
        "battles": 0, "wins": 0, "losses": 0, "draws": 0,
        "turns": [], "moves": 0, "switches": 0,
        "voluntary_switches": 0, "forced_switches": 0,
        "action_requests": 0,
    })

    for row in battles:
        gen = int(row["generation"])
        a1, a2 = row["agent_1"], row["agent_2"]
        result = row["result_agent_1"]
        turns = int(float(row["turns"])) if row.get("turns") else None

        for agent in (a1, a2):
            key = (gen, agent)
            stats[key]["battles"] += 1
            if turns is not None:
                stats[key]["turns"].append(turns)

        if result == "win":
            stats[(gen, a1)]["wins"] += 1
            stats[(gen, a2)]["losses"] += 1
        elif result == "loss":
            stats[(gen, a1)]["losses"] += 1
            stats[(gen, a2)]["wins"] += 1
        else:
            stats[(gen, a1)]["draws"] += 1
            stats[(gen, a2)]["draws"] += 1

    for row in actions:
        key = (int(row["generation"]), row["agent"])
        stats[key]["action_requests"] += 1
        if row["action_type"] == "move":
            stats[key]["moves"] += 1
        elif row["action_type"] == "switch":
            stats[key]["switches"] += 1
        if row.get("voluntary_switch", "").lower() == "true":
            stats[key]["voluntary_switches"] += 1
        if row.get("forced_switch", "").lower() == "true":
            stats[key]["forced_switches"] += 1

    rows = []
    for (generation, agent), s in sorted(stats.items()):
        n = s["battles"]
        decisive = s["wins"] + s["losses"]
        action_n = s["action_requests"]
        rows.append({
            "generation": generation,
            "agent": agent,
            "battles": n,
            "wins": s["wins"],
            "losses": s["losses"],
            "draws": s["draws"],
            "win_rate_all": round(s["wins"] / n, 6) if n else 0.0,
            "win_rate_decisive": round(s["wins"] / decisive, 6) if decisive else 0.0,
            "mean_battle_turns": round(mean(s["turns"]), 3) if s["turns"] else "",
            "action_requests": action_n,
            "moves": s["moves"],
            "switches": s["switches"],
            "switch_rate": round(s["switches"] / action_n, 6) if action_n else 0.0,
            "voluntary_switches": s["voluntary_switches"],
            "forced_switches": s["forced_switches"],
        })

    _write_csv(
        run_dir / "agent_summary.csv",
        rows,
        [
            "generation", "agent", "battles", "wins", "losses", "draws",
            "win_rate_all", "win_rate_decisive", "mean_battle_turns",
            "action_requests", "moves", "switches", "switch_rate",
            "voluntary_switches", "forced_switches",
        ],
    )
    return rows


def create_plots(run_dir: Path, agent_rows: list[dict]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    figure_dir = run_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    generated = []

    generations = sorted({int(r["generation"]) for r in agent_rows})
    agents = sorted({r["agent"] for r in agent_rows})

    # Win-rate plot: one line per agent across generations.
    fig, ax = plt.subplots(figsize=(8, 5))
    for agent in agents:
        values = []
        for gen in generations:
            match = next((r for r in agent_rows if r["agent"] == agent and int(r["generation"]) == gen), None)
            values.append(float(match["win_rate_all"]) if match else float("nan"))
        ax.plot(generations, values, marker="o", label=agent)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1)
    ax.set_title("Agent win rate by generation")
    ax.legend()
    fig.tight_layout()
    path = figure_dir / "agent_win_rate.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(path)

    # Switch-rate plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    for agent in agents:
        values = []
        for gen in generations:
            match = next((r for r in agent_rows if r["agent"] == agent and int(r["generation"]) == gen), None)
            values.append(float(match["switch_rate"]) if match else float("nan"))
        ax.plot(generations, values, marker="o", label=agent)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Switch requests / action requests")
    ax.set_ylim(0, 1)
    ax.set_title("Switch rate by generation")
    ax.legend()
    fig.tight_layout()
    path = figure_dir / "switch_rate.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(path)

    return generated


def analyze_run(run_dir: Path, make_plots: bool = True) -> list[dict]:
    agent_rows = build_agent_summary(run_dir)
    if make_plots:
        create_plots(run_dir, agent_rows)
    return agent_rows
