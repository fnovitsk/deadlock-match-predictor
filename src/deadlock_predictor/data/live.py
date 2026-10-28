"""Games in progress, from the API's `/v1/matches/active`.

For each game the API gives each team's current net worth and objectives mask, every
player's hero and the start time, but no kills, levels, damage, mid-boss or rank. So live
predictions use a model trained on just these inputs (`models.gbm.LIVE_COLUMNS`).

The objectives mask has bit i set while objective id i (`data.match.OBJECTIVES`) is still
standing; that held for every objective in 3,000 dump matches.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from deadlock_predictor.data.dumps import USER_AGENT
from deadlock_predictor.data.match import OBJECTIVES
from deadlock_predictor.features.snapshots import OBJECTIVE_GROUPS

ACTIVE_URL = "https://api.deadlock-api.com/v1/matches/active"
OBJ_COLUMNS = [f"obj_{g}_diff" for g in OBJECTIVE_GROUPS.values()]
# `start_time` is ~210 s before the game clock starts (the pre-game). Measured by
# `live.score`'s clock check on 413 finished games: live team net worth matched the
# post-game snapshots within 6% at this offset, against 25% without it.
PREGAME_S = 210


def _group(name: str) -> str | None:
    # TitanShieldGenerator before Titan (shared prefix), as in features.snapshots.
    for prefix, group in sorted(OBJECTIVE_GROUPS.items(), key=lambda kv: -len(kv[0])):
        if name.startswith(prefix):
            return group
    return None  # Core


GROUP_OF = {i: _group(name) for i, name in OBJECTIVES.items()}


def fetch_active(timeout: float = 30) -> list[dict]:
    """Ranked 6v6 games in progress."""
    response = requests.get(ACTIVE_URL, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return [
        g
        for g in response.json()
        if g.get("match_mode_parsed") == "Ranked"
        and g.get("game_mode_parsed") == "KECitadelGameModeNormal"
    ]


def objectives_lost(mask: int) -> dict[str, int]:
    """How many objectives of each group a team has lost, from its objectives mask."""
    lost = dict.fromkeys(OBJECTIVE_GROUPS.values(), 0)
    for i, group in GROUP_OF.items():
        if group and not mask >> i & 1:
            lost[group] += 1
    return lost


def game_time(now: float, start_time: float) -> float:
    """Seconds on the game clock: time since `start_time`, minus the pre-game."""
    return max(0.0, now - start_time - PREGAME_S)


def live_state(game: dict, now: float | None = None) -> dict:
    """One row for a game in progress: the raw values and the model inputs.

    Signs follow training: diffs are Team0 − Team1, and losing your own objective counts
    against you, so obj_<g>_diff = (Team1's losses) − (Team0's losses).
    """
    now = time.time() if now is None else now
    teams = {s["team"]: s for s in game["team_stats"]}
    lost0, lost1 = (objectives_lost(teams[i]["objectives_mask"]) for i in (0, 1))
    heroes = {i: sorted(p["hero_id"] for p in game["players"] if p["team"] == i) for i in (0, 1)}
    return {
        "match_id": game["match_id"],
        "polled_at": pd.Timestamp(now, unit="s", tz="UTC"),
        "start_time": pd.Timestamp(game["start_time"], unit="s", tz="UTC"),
        "net_worth_team0": teams[0]["net_worth"],
        "net_worth_team1": teams[1]["net_worth"],
        "objectives_mask_team0": teams[0]["objectives_mask"],
        "objectives_mask_team1": teams[1]["objectives_mask"],
        "heroes_team0": heroes[0],
        "heroes_team1": heroes[1],
        "abandoned": sum(bool(p.get("abandoned")) for p in game["players"]),
        "t": game_time(now, game["start_time"]),
        "net_worth_diff": float(teams[0]["net_worth"] - teams[1]["net_worth"]),
        **{f"obj_{g}_diff": float(lost1[g] - lost0[g]) for g in OBJECTIVE_GROUPS.values()},
    }


def heroes_frame(states: pd.DataFrame) -> pd.DataFrame:
    """Hero picks in `heroes.parquet` form (match_id, hero_id, sign) for `add_heroes`."""
    games = states.drop_duplicates("match_id")
    rows = [
        (g.match_id, h, sign)
        for g in games.itertuples()
        for heroes, sign in ((g.heroes_team0, 1), (g.heroes_team1, -1))
        for h in heroes
    ]
    return pd.DataFrame(rows, columns=["match_id", "hero_id", "sign"]).astype(
        {"hero_id": np.int64, "sign": np.int8}
    )
