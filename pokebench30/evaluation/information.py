from __future__ import annotations

from typing import Any


def team_snapshot(team: Any) -> tuple[int, int, int, float]:
    """Return team size/known count, remaining, fainted and summed HP fraction.

    The meaning of the first value depends on the supplied view:
    - for a player's own team, it is the full team size;
    - for opponent_team, it is only the number of revealed opponents.
    """
    if not team:
        return 0, 0, 0, 0.0

    values = list(team.values()) if hasattr(team, "values") else list(team)
    known = len(values)
    fainted = sum(bool(getattr(mon, "fainted", False)) for mon in values)
    remaining = known - fainted
    hp_sum = sum(
        float(getattr(mon, "current_hp_fraction", 0.0) or 0.0)
        for mon in values
    )
    return known, remaining, fainted, round(hp_sum, 6)


def final_information_snapshot(p1_battle: Any, p2_battle: Any) -> dict:
    """Separate evaluator ground truth from each player's partial observation.

    This function is intentionally simulator-agnostic. It expects battle-like
    objects exposing ``team`` and ``opponent_team``.

    Ground truth is derived post-battle from each player's OWN team.
    Partial observation is derived from each player's OWN opponent_team view.
    """
    p1_true = team_snapshot(getattr(p1_battle, "team", {}))
    p2_true = team_snapshot(getattr(p2_battle, "team", {}))

    p1_seen = team_snapshot(getattr(p1_battle, "opponent_team", {}))
    p2_seen = team_snapshot(getattr(p2_battle, "opponent_team", {}))

    (
        p1_team_size_true,
        p1_remaining_true,
        p1_fainted_true,
        p1_hp_sum_true,
    ) = p1_true
    (
        p2_team_size_true,
        p2_remaining_true,
        p2_fainted_true,
        p2_hp_sum_true,
    ) = p2_true

    (
        p1_opponent_revealed,
        p1_opponent_revealed_remaining,
        p1_opponent_revealed_fainted,
        p1_opponent_revealed_hp_sum,
    ) = p1_seen
    (
        p2_opponent_revealed,
        p2_opponent_revealed_remaining,
        p2_opponent_revealed_fainted,
        p2_opponent_revealed_hp_sum,
    ) = p2_seen

    p1_reveal_fraction = (
        p1_opponent_revealed / p2_team_size_true
        if p2_team_size_true
        else 0.0
    )
    p2_reveal_fraction = (
        p2_opponent_revealed / p1_team_size_true
        if p1_team_size_true
        else 0.0
    )

    return {
        # Evaluator-only ground truth.
        "agent_1_team_size_true": p1_team_size_true,
        "agent_2_team_size_true": p2_team_size_true,
        "agent_1_remaining_true": p1_remaining_true,
        "agent_2_remaining_true": p2_remaining_true,
        "agent_1_fainted_true": p1_fainted_true,
        "agent_2_fainted_true": p2_fainted_true,
        "agent_1_hp_sum_true": p1_hp_sum_true,
        "agent_2_hp_sum_true": p2_hp_sum_true,

        # Final partial-information view of each player.
        "agent_1_opponent_revealed": p1_opponent_revealed,
        "agent_2_opponent_revealed": p2_opponent_revealed,
        "agent_1_opponent_revealed_remaining": p1_opponent_revealed_remaining,
        "agent_2_opponent_revealed_remaining": p2_opponent_revealed_remaining,
        "agent_1_opponent_revealed_fainted": p1_opponent_revealed_fainted,
        "agent_2_opponent_revealed_fainted": p2_opponent_revealed_fainted,
        "agent_1_opponent_revealed_hp_sum": p1_opponent_revealed_hp_sum,
        "agent_2_opponent_revealed_hp_sum": p2_opponent_revealed_hp_sum,
        "agent_1_opponent_reveal_fraction": round(p1_reveal_fraction, 6),
        "agent_2_opponent_reveal_fraction": round(p2_reveal_fraction, 6),
    }
