"""Group a match's deaths into teamfights.

A death joins an open fight if it happens within `gap_s` of that fight's last death and
within `radius` world units of the fight's centre (the mean position of its deaths so
far); otherwise it opens a new fight. Several fights can be open at once, so fights in
different lanes stay apart. Groups with fewer than `min_deaths` deaths are skirmishes and
are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from deadlock_predictor.viz.curve import TEAM_NAMES

COLUMNS = [
    "fight",
    "start",
    "end",
    "x",
    "y",
    "deaths_team0",
    "deaths_team1",
    "winner",
    "label",
    "victims",
    "killers",
]


def fights(
    deaths: pd.DataFrame, gap_s: float = 20, radius: float = 3000, min_deaths: int = 3
) -> pd.DataFrame:
    """One row per fight: start/end time, centre, deaths per side, the winner (the side
    that lost fewer players; None if even), a label like "Fight 4-1 (Hidden King)", and the
    slots of the victims and of the credited killers."""
    groups: list[list] = []  # each: list of death rows
    for d in deaths.sort_values("t").itertuples():
        best, best_dist = None, np.inf
        for g in groups:
            if d.t - g[-1].t > gap_s:
                continue
            cx = np.nanmean([e.x for e in g])
            cy = np.nanmean([e.y for e in g])
            dist = np.hypot(d.x - cx, d.y - cy)
            dist = 0.0 if np.isnan(dist) else dist  # no position: join on time alone
            if dist <= radius and dist < best_dist:
                best, best_dist = g, dist
        if best is None:
            groups.append([d])
        else:
            best.append(d)
    rows = []
    for g in groups:
        if len(g) < min_deaths:
            continue
        lost = {team: sum(e.victim_team == team for e in g) for team in ("Team0", "Team1")}
        kills0, kills1 = lost["Team1"], lost["Team0"]  # Team0's kills are Team1's deaths
        winner = "Team0" if kills0 > kills1 else "Team1" if kills1 > kills0 else None
        score = f"{max(kills0, kills1)}-{min(kills0, kills1)}"
        label = f"Fight {score} ({TEAM_NAMES[winner]})" if winner else f"Fight {score} (even)"
        rows.append(
            {
                "start": g[0].t,
                "end": g[-1].t,
                "x": float(np.nanmean([e.x for e in g])),
                "y": float(np.nanmean([e.y for e in g])),
                "deaths_team0": lost["Team0"],
                "deaths_team1": lost["Team1"],
                "winner": winner,
                "label": label,
                "victims": [e.victim for e in g],
                "killers": sorted({int(e.killer) for e in g if pd.notna(e.killer)}),
            }
        )
    out = pd.DataFrame(rows, columns=COLUMNS[1:]).sort_values("start", ignore_index=True)
    out.insert(0, "fight", np.arange(1, len(out) + 1))
    return out
