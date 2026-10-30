"""Everything the API knows about one finished match, as tidy frames.

The dump keeps 10 stats per player snapshot. The API's match metadata
(`/v1/matches/{id}/metadata`, `data.match.fetch_match`) has much more:
- every death, with its second, the killer and both positions;
- ~50 stats per player snapshot.

It also has item purchases, buff pickups and positions once a second (`match_paths`);
parsers for those were removed with the views that used them (git history, M7/M8).

`fetch_detail` caches the metadata in `data/matches/`. The parsers below turn it into
frames with teams as 'Team0'/'Team1' and times in seconds of game time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from deadlock_predictor.data.match import ROOT, TEAMS, fetch_match

CACHE_DIR = ROOT / "data" / "matches"


def fetch_detail(match_id: int, cache_dir: Path = CACHE_DIR) -> dict:
    """The API's `match_info` for one match, from the cache if it's there."""
    path = cache_dir / f"{int(match_id)}.json"
    if path.exists():
        return json.loads(path.read_text())
    info = fetch_match(match_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(info))
    tmp.replace(path)
    return info


def players(info: dict) -> pd.DataFrame:
    """One row per player: slot, team, hero_id, lane, account_id. Team0 first."""
    rows = [
        {
            "slot": p["player_slot"],
            "team": TEAMS[p["team"]],
            "hero_id": p["hero_id"],
            "lane": p.get("assigned_lane"),
            "account_id": p.get("account_id"),
        }
        for p in info["players"]
    ]
    return pd.DataFrame(rows).sort_values(["team", "slot"], ignore_index=True)


def player_stats(info: dict) -> pd.DataFrame:
    """One row per player and stat snapshot (including the final one, at the end of the
    match): slot, team, t, and every numeric stat the API reports."""
    rows = []
    for p in info["players"]:
        for s in p.get("stats") or []:
            row = {k: v for k, v in s.items() if isinstance(v, int | float)}
            row["t"] = float(row.pop("time_stamp_s"))
            rows.append({"slot": p["player_slot"], "team": TEAMS[p["team"]], **row})
    return pd.DataFrame(rows).sort_values(["slot", "t"], ignore_index=True)


def deaths(info: dict) -> pd.DataFrame:
    """Every death: t, victim and killer (slot and team; killer None if no enemy player got
    the kill), positions, and how long the victim was dead."""
    team_of = {p["player_slot"]: TEAMS[p["team"]] for p in info["players"]}
    rows = []
    for p in info["players"]:
        victim_team = TEAMS[p["team"]]
        for d in p.get("death_details") or []:
            killer = d.get("killer_player_slot")
            killer_team = team_of.get(killer)
            if killer_team in (None, victim_team):
                killer, killer_team = None, None
            rows.append(
                {
                    "t": float(d["game_time_s"]),
                    "victim": p["player_slot"],
                    "victim_team": victim_team,
                    "killer": killer,
                    "killer_team": killer_team,
                    "x": d.get("death_pos", {}).get("x", np.nan),
                    "y": d.get("death_pos", {}).get("y", np.nan),
                    "dead_s": float(d.get("death_duration_s") or 0),
                }
            )
    columns = ["t", "victim", "victim_team", "killer", "killer_team", "x", "y", "dead_s"]
    return pd.DataFrame(rows, columns=columns).sort_values("t", ignore_index=True)
