"""Animated replay of a match on the real minimap (Plotly frames, played in the browser).

Each frame (every `step_s` seconds of game time) shows every living player as a circle in
their hero's colour with a ring in their team's colour, sized by health, with a short
trail; recent deaths as ✕; and every building, greyed out once destroyed.

Building positions: the API's asset positions don't line up with the minimap, so they are
estimated from the data instead (`estimate_buildings`): when a building falls, most of
the attacking team stands at it. Run `python -m deadlock_predictor.viz.map_replay` to
re-estimate them from the cached matches (`data/matches/`) into `configs/buildings.yaml`.
"""

from __future__ import annotations

import argparse
import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

from deadlock_predictor.data import match_detail as md
from deadlock_predictor.data.assets import map_info
from deadlock_predictor.data.match import OBJECTIVES, ROOT, TEAMS
from deadlock_predictor.viz.curve import EVENT_LABELS, EVENT_NAMES, TEAM_NAMES
from deadlock_predictor.viz.explorer import Match, clock
from deadlock_predictor.viz.theme import MUTED, TEAM_COLORS, TEXT, styled
from deadlock_predictor.viz.timeline import _group

BUILDINGS_PATH = ROOT / "configs" / "buildings.yaml"
MAX_STEP = 1500  # world units in one second: more is a respawn or teleport
DEATH_SHOWN_S = 10


def _densest(xy: np.ndarray, radius: float) -> tuple[np.ndarray, int]:
    """The mean of the largest group of points within `radius` of one of them."""
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    best = (d < radius).sum(1).argmax()
    near = d[best] < radius
    return xy[near].mean(0), int(near.sum())


def estimate_buildings(infos, window_s: float = 10) -> pd.DataFrame:
    """Building positions from where buildings fall, over many matches' metadata.

    Per destroyed building: the attackers' positions over its last `window_s` seconds, and
    the centre of their densest group (attackers stand at the building while hitting it).
    Per building: the centre of the densest group of those per-match estimates, which
    ignores matches where the attackers' main group was elsewhere."""
    estimates = []
    for info in infos:
        paths = md.paths(info)
        if paths.empty:
            continue
        for o in info.get("objectives") or []:
            t = o.get("destroyed_time_s") or 0
            name = OBJECTIVES.get(o.get("team_objective_id"))
            if t <= 0 or name is None or name == "Core":
                continue
            owner = TEAMS[o["team"]]
            scorer = "Team1" if owner == "Team0" else "Team0"
            near = paths[
                (paths.t > t - window_s)
                & (paths.t <= t)
                & (paths.team == scorer)
                & (paths.health > 0)
            ]
            if len(near) < 5:
                continue
            (x, y), _ = _densest(near[["x", "y"]].to_numpy(), 1200)
            estimates.append({"owner": owner, "name": name, "x": x, "y": y})
    out = []
    for (owner, name), g in pd.DataFrame(estimates).groupby(["owner", "name"]):
        (x, y), agree = _densest(g[["x", "y"]].to_numpy(), 1000)
        out.append(
            {
                "owner": owner,
                "name": name,
                "x": round(x),
                "y": round(y),
                "agree": agree,
                "of": len(g),
            }
        )
    return pd.DataFrame(out)


@cache
def buildings() -> pd.DataFrame:
    """owner, name, x, y of every building (from configs/buildings.yaml)."""
    data = yaml.safe_load(BUILDINGS_PATH.read_text()) if BUILDINGS_PATH.exists() else {}
    rows = [
        {"owner": owner, "name": name, "x": float(xy[0]), "y": float(xy[1])}
        for owner, names in (data or {}).items()
        if owner in TEAM_NAMES
        for name, xy in names.items()
    ]
    return pd.DataFrame(rows, columns=["owner", "name", "x", "y"])


class _Tracks:
    """Every player's x, y and health at every second, as players × seconds arrays (NaN
    where unknown), so a frame is an array slice rather than a table filter."""

    def __init__(self, m: Match):
        p = m.players.reset_index(drop=True)
        self.hero, self.color, self.team = p.hero.to_numpy(), p.color.to_numpy(), p.team.to_numpy()
        row = {slot: i for i, slot in enumerate(p.slot)}
        n_t = int(m.paths.t.max()) + 1
        self.x, self.y, self.health = (np.full((len(p), n_t), np.nan) for _ in range(3))
        i = m.paths.slot.map(row).to_numpy()
        t = m.paths.t.to_numpy(int)
        ok = ~np.isnan(i.astype(float))
        i, t = i[ok].astype(int), t[ok]
        self.x[i, t] = m.paths.x.to_numpy()[ok]
        self.y[i, t] = m.paths.y.to_numpy()[ok]
        self.health[i, t] = m.paths.health.to_numpy()[ok]

    def at(self, t: int):
        t = min(t, self.x.shape[1] - 1)
        return self.x[:, t], self.y[:, t], self.health[:, t]

    def trails(self, t: int, trail_s: int) -> tuple[list, list]:
        """The last `trail_s` seconds of every living player's path (every 2 s), broken at
        respawns and teleports, as one None-separated line."""
        lo, hi = max(0, t - trail_s), min(t, self.x.shape[1] - 1) + 1
        xs, ys = [], []
        for x, y, h in zip(
            self.x[:, lo:hi:2], self.y[:, lo:hi:2], self.health[:, lo:hi:2], strict=True
        ):
            keep = (h > 0) & ~np.isnan(x)
            x, y = x[keep], y[keep]
            if len(x) < 2:
                continue
            jump = np.concatenate([[False], np.hypot(np.diff(x), np.diff(y)) > 2 * MAX_STEP])
            for xi, yi, j in zip(x.round(), y.round(), jump, strict=True):
                if j:
                    xs.append(None)
                    ys.append(None)
                xs.append(xi)
                ys.append(yi)
            xs.append(None)
            ys.append(None)
        return xs, ys


def _frame_data(
    m: Match,
    tracks: _Tracks,
    t: int,
    trail_s: int,
    b: pd.DataFrame,
    fallen: dict,
    labels: list[str],
) -> list[dict]:
    """The traces of one frame (plain dicts: much faster than validated Plotly objects):
    trails, players, recent deaths, standing and fallen buildings."""
    tx, ty = tracks.trails(t, trail_s)
    x, y, h = tracks.at(t)
    alive = (h > 0) & ~np.isnan(x)
    dead = m.deaths[(m.deaths.t <= t) & (m.deaths.t > t - DEATH_SHOWN_S)]
    down = np.array(
        [fallen.get((o, n), np.inf) <= t for o, n in zip(b.owner, b.name, strict=True)], dtype=bool
    )
    return [
        {
            "type": "scatter",
            "x": tx,
            "y": ty,
            "mode": "lines",
            "hoverinfo": "skip",
            "line": {"color": "rgba(227,230,236,0.35)", "width": 1.5},
        },
        {
            "type": "scatter",
            "x": x[alive].round().tolist(),
            "y": y[alive].round().tolist(),
            "mode": "markers+text",
            "text": tracks.hero[alive].tolist(),
            "textposition": "top center",
            "textfont": {"size": 10, "color": TEXT},
            "marker": {
                "size": (10 + 10 * h[alive] / 100).round(1).tolist(),
                "color": tracks.color[alive].tolist(),
                "line": {"color": [TEAM_COLORS[team] for team in tracks.team[alive]], "width": 3},
            },
            "customdata": h[alive].round().tolist(),
            "hovertemplate": "%{text}: %{customdata:.0f}% health<extra></extra>",
        },
        {
            "type": "scatter",
            "x": dead.x.round().tolist(),
            "y": dead.y.round().tolist(),
            "mode": "markers",
            "hoverinfo": "skip",
            "marker": {
                "symbol": "x",
                "size": 14,
                "line": {"width": 1, "color": TEXT},
                "color": [TEAM_COLORS[v] for v in dead.victim_team],
            },
        },
        {
            "type": "scatter",
            "x": b.x[~down].tolist(),
            "y": b.y[~down].tolist(),
            "mode": "markers+text",
            "hoverinfo": "skip",
            "text": [lab for lab, d in zip(labels, down, strict=True) if not d],
            "textposition": "bottom center",
            "textfont": {"size": 9, "color": MUTED},
            "marker": {
                "symbol": "square",
                "size": 11,
                "line": {"width": 1, "color": TEXT},
                "color": [TEAM_COLORS[o] for o in b.owner[~down]],
            },
        },
        {
            "type": "scatter",
            "x": b.x[down].tolist(),
            "y": b.y[down].tolist(),
            "mode": "markers",
            "hoverinfo": "skip",
            "marker": {"symbol": "square-x-open", "size": 11, "color": MUTED},
        },
    ]


def replay_figure(
    m: Match, step_s: int = 5, start_s: float = 0, trail_s: float = 20, height: int = 760
) -> go.Figure:
    """The animated replay; it opens at `start_s`."""
    info = map_info()
    r = info.radius
    b = buildings()
    ev = m.events.dropna(subset=["name"])
    fallen = {(o, n): t for o, n, t in zip(ev.owner, ev.name, ev.t, strict=True)}
    times = np.arange(0, int(m.duration_s) + 1, step_s)
    if m.paths.empty:
        fig = go.Figure()
        fig.add_annotation(text="No position data for this match", showarrow=False)
        return styled(fig, 200)
    tracks = _Tracks(m)
    labels = [EVENT_LABELS[EVENT_NAMES[_group(n)]] for n in b.name]
    frames = [
        {
            "name": str(int(t)),
            "layout": {"title": {"text": clock(t)}},
            "data": _frame_data(m, tracks, int(t), int(trail_s), b, fallen, labels),
        }
        for t in times
    ]
    first = min(int(np.searchsorted(times, start_s)), len(times) - 1)
    fig = go.Figure({"data": frames[first]["data"], "frames": frames})
    if info.dark:
        fig.add_layout_image(
            source=info.dark,
            xref="x",
            yref="y",
            x=-r,
            y=r,
            sizex=2 * r,
            sizey=2 * r,
            xanchor="left",
            yanchor="top",
            sizing="stretch",
            layer="below",
        )
    play = {
        "frame": {"duration": 120, "redraw": False},
        "transition": {"duration": 0},
        "fromcurrent": True,
        "mode": "immediate",
    }
    fast = {**play, "frame": {"duration": 40, "redraw": False}}
    pause = {
        "frame": {"duration": 0, "redraw": False},
        "mode": "immediate",
        "transition": {"duration": 0},
    }
    fig.update_layout(
        title={"text": clock(times[first])},
        showlegend=False,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.02,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"t": 0, "r": 10},
                "showactive": False,
                "bgcolor": "#171b23",
                "bordercolor": "#262c38",
                "font": {"color": TEXT},
                "buttons": [
                    {"label": "▶ Play", "method": "animate", "args": [None, play]},
                    {"label": "⏩ Fast", "method": "animate", "args": [None, fast]},
                    {"label": "⏸ Pause", "method": "animate", "args": [[None], pause]},
                ],
            }
        ],
        sliders=[
            {
                "active": first,
                "x": 0.25,
                "len": 0.75,
                "y": -0.02,
                "yanchor": "top",
                "currentvalue": {"prefix": "time ", "font": {"color": TEXT}},
                "font": {"color": MUTED, "size": 10},
                "bgcolor": "#262c38",
                "tickcolor": "#262c38",
                "steps": [
                    {
                        "label": clock(t) if t % 300 == 0 else "",
                        "method": "animate",
                        "args": [[str(int(t))], pause],
                    }
                    for t in times
                ],
            }
        ],
        margin={"l": 10, "r": 10, "t": 50, "b": 90},
    )
    fig.update_xaxes(range=[-r, r], visible=False)
    fig.update_yaxes(range=[-r, r], visible=False, scaleanchor="x")
    return styled(fig, height)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-estimate building positions")
    parser.add_argument("--matches", type=Path, default=md.CACHE_DIR)
    args = parser.parse_args()
    infos = [json.loads(p.read_text()) for p in sorted(args.matches.glob("*.json"))]
    est = estimate_buildings(infos)
    print(est.to_string(index=False))
    data = {
        owner: {n: [int(x), int(y)] for n, x, y in zip(g.name, g.x, g.y, strict=True)}
        for owner, g in est.groupby("owner")
    }
    header = (
        "# Building positions (world units, y up), from viz.map_replay.estimate_buildings\n"
        f"# over {len(infos)} cached matches: where the attackers stand when each building falls.\n"
        "# Regenerate after a map change: python -m deadlock_predictor.viz.map_replay\n"
    )
    BUILDINGS_PATH.write_text(header + yaml.safe_dump(data, sort_keys=True), newline="\n")
    print("wrote", BUILDINGS_PATH)


if __name__ == "__main__":
    main()
