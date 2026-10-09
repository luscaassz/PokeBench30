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
        "opponent_reveal_fractions": [],
        "switch_loop_events": 0,
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

        try:
            stats[(gen, a1)]["opponent_reveal_fractions"].append(
                float(row.get("agent_1_opponent_reveal_fraction", 0.0) or 0.0)
            )
            stats[(gen, a2)]["opponent_reveal_fractions"].append(
                float(row.get("agent_2_opponent_reveal_fraction", 0.0) or 0.0)
            )
        except (TypeError, ValueError):
            pass

        if result == "win":
            stats[(gen, a1)]["wins"] += 1
            stats[(gen, a2)]["losses"] += 1
        elif result == "loss":
            stats[(gen, a1)]["losses"] += 1
            stats[(gen, a2)]["wins"] += 1
        else:
            stats[(gen, a1)]["draws"] += 1
            stats[(gen, a2)]["draws"] += 1

    last_voluntary_switch = {}

    actions = sorted(
        actions,
        key=lambda row: (
            int(row["generation"]),
            row["agent"],
            row["battle_tag"],
            int(row.get("request_index", 0) or 0),
        ),
    )

    for row in actions:
        key = (int(row["generation"]), row["agent"])
        stats[key]["action_requests"] += 1
        if row["action_type"] == "move":
            stats[key]["moves"] += 1
        elif row["action_type"] == "switch":
            stats[key]["switches"] += 1
            if row.get("voluntary_switch", "").lower() == "true":
                stats[key]["voluntary_switches"] += 1

                battle_key = (
                    int(row["generation"]),
                    row["agent"],
                    row["battle_tag"],
                )

                current_switch = {
                    "from": row.get("active_pokemon", ""),
                    "to": row.get("action", ""),
                    "turn": int(
                        float(row.get("turn", 0) or 0)
                    ),
                }

                previous_switch = last_voluntary_switch.get(
                    battle_key
                )

                if previous_switch is not None:
                    reversed_switch = (
                        previous_switch["from"]
                        == current_switch["to"]
                        and previous_switch["to"]
                        == current_switch["from"]
                    )

                    close_in_time = (
                        current_switch["turn"]
                        - previous_switch["turn"]
                        <= 2
                    )

                    if (
                        reversed_switch
                        and close_in_time
                    ):
                        stats[key]["switch_loop_events"] += 1

                last_voluntary_switch[
                    battle_key
                ] = current_switch

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
            "voluntary_switch_rate": round(s["voluntary_switches"] / action_n, 6) if action_n else 0.0,
            "forced_switch_rate": round(s["forced_switches"] / action_n, 6) if action_n else 0.0,
            "voluntary_switches": s["voluntary_switches"],
            "forced_switches": s["forced_switches"],
            "switch_loop_events": s["switch_loop_events"],
            "switch_loop_rate": round(
                s["switch_loop_events"]
                / s["voluntary_switches"],
                6,
            )
            if s["voluntary_switches"]
            else 0.0,
            "mean_opponent_reveal_fraction": round(
                mean(s["opponent_reveal_fractions"]), 6
            ) if s["opponent_reveal_fractions"] else 0.0,
        })

    _write_csv(
        run_dir / "agent_summary.csv",
        rows,
        [
            "generation", "agent", "battles", "wins", "losses", "draws",
            "win_rate_all", "win_rate_decisive", "mean_battle_turns",
            "action_requests", "moves", "switches", "switch_rate",
            "voluntary_switch_rate", "forced_switch_rate",
            "voluntary_switches",
            "forced_switches",

            "switch_loop_events",
            "switch_loop_rate",

            "mean_opponent_reveal_fraction",
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

    def value(agent, gen, field):
        match = next(
            (
                r for r in agent_rows
                if r["agent"] == agent and int(r["generation"]) == gen
            ),
            None,
        )
        return float(match[field]) if match else float("nan")

    # Win rate: bars for a single generation; lines for longitudinal runs.
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(generations) == 1:
        gen = generations[0]
        vals = [value(agent, gen, "win_rate_all") for agent in agents]
        ax.bar(agents, vals)
        ax.set_title(f"Agent win rate — Generation {gen}")
        ax.tick_params(axis="x", rotation=20)
    else:
        for agent in agents:
            vals = [value(agent, gen, "win_rate_all") for gen in generations]
            ax.plot(generations, vals, marker="o", label=agent)
        ax.set_xticks(generations)
        ax.set_xlabel("Generation")
        ax.set_title("Agent win rate by generation")
        ax.legend()
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = figure_dir / "agent_win_rate.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(path)

    # Total switch rate (kept for backwards compatibility).
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(generations) == 1:
        gen = generations[0]
        vals = [value(agent, gen, "switch_rate") for agent in agents]
        ax.bar(agents, vals)
        ax.set_title(f"Total switch rate — Generation {gen}")
        ax.tick_params(axis="x", rotation=20)
    else:
        for agent in agents:
            vals = [value(agent, gen, "switch_rate") for gen in generations]
            ax.plot(generations, vals, marker="o", label=agent)
        ax.set_xticks(generations)
        ax.set_xlabel("Generation")
        ax.set_title("Total switch rate by generation")
        ax.legend()
    ax.set_ylabel("All switches / action requests")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = figure_dir / "switch_rate.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(path)

    # Strategic switching: forced switches removed.
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(generations) == 1:
        gen = generations[0]
        vals = [value(agent, gen, "voluntary_switch_rate") for agent in agents]
        ax.bar(agents, vals)
        ax.set_title(f"Voluntary switch rate — Generation {gen}")
        ax.tick_params(axis="x", rotation=20)
    else:
        for agent in agents:
            vals = [
                value(agent, gen, "voluntary_switch_rate")
                for gen in generations
            ]
            ax.plot(generations, vals, marker="o", label=agent)
        ax.set_xticks(generations)
        ax.set_xlabel("Generation")
        ax.set_title("Voluntary switch rate by generation")
        ax.legend()
    ax.set_ylabel("Voluntary switches / action requests")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = figure_dir / "voluntary_switch_rate.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(path)

    return generated


def analyze_run(run_dir: Path, make_plots: bool = True) -> list[dict]:
    agent_rows = build_agent_summary(run_dir)
    if make_plots:
        create_plots(run_dir, agent_rows)
    return agent_rows
