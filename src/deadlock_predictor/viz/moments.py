"""Key moments: the biggest swings in win probability, and what caused them.

The curve (`viz.timeline.event_points`) is a sequence of points. Each step between two
consecutive points changes P, and the final model's grouped contributions say which
inputs moved it. Steps within `window_s` of each other are merged into one moment (a
fight and the Walker it opened up are one story), then moments are ranked by how much
they moved P.
"""

from __future__ import annotations

import pandas as pd

from deadlock_predictor.viz.curve import GROUPS

DRIFT = "souls and stats (between snapshots)"


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def steps(points: pd.DataFrame, fights: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each step between consecutive curve points: t (its end), dp, the change in each
    contribution group, and its label (the trigger, or the fight it falls in)."""
    pts = points.sort_values("t").reset_index(drop=True)
    groups = [g for g in GROUPS if g in pts.columns]
    out = pd.DataFrame(
        {
            "t": pts.t.iloc[1:].to_numpy(),
            "dp": pts.p_final.diff().iloc[1:].to_numpy(),
            # A stat snapshot isn't an event: its change is drift between snapshots.
            "label": pts.trigger.iloc[1:].fillna("").replace("stat snapshot", "").to_numpy(),
        }
    )
    for g in groups:
        out[g] = pts[g].diff().iloc[1:].to_numpy()
    if fights is not None and len(fights):
        for f in fights.itertuples():
            inside = (out.t >= f.start - 1) & (out.t <= f.end) & (out.label == "")
            out.loc[inside, "label"] = f.label
    return out


def key_moments(
    points: pd.DataFrame, fights: pd.DataFrame | None = None, n: int = 8, window_s: float = 60
) -> pd.DataFrame:
    """The `n` biggest moments, biggest first: start, end, dp (change in P(Team0 wins)),
    team (who it helped), causes (the groups that moved it most), labels and a text."""
    s = steps(points, fights)
    if s.empty:
        return pd.DataFrame(columns=["start", "end", "dp", "team", "causes", "labels", "text"])
    groups = [g for g in GROUPS if g in s.columns]
    moment, start = [], None
    for t in s.t:
        if start is None or t - start > window_s:
            start = t
        moment.append(start)
    s["moment"] = moment
    rows = []
    for _, m in s.groupby("moment"):
        dp = float(m.dp.sum())
        labels = list(dict.fromkeys(label for label in m.label if label)) or [DRIFT]
        moved = m[groups].sum()
        # The groups that pushed in the moment's direction, strongest first.
        toward = moved[moved * (1 if dp >= 0 else -1) > 0].abs().sort_values(ascending=False)
        rows.append(
            {
                "start": float(m.t.min()),
                "end": float(m.t.max()),
                "dp": dp,
                "team": "Team0" if dp >= 0 else "Team1",
                "causes": list(toward.index[:2]),
                "labels": labels,
            }
        )
    out = pd.DataFrame(rows)
    out = out.reindex(out.dp.abs().sort_values(ascending=False).index).head(n)
    out["text"] = [
        f"{_clock(r.start)}  {'; '.join(r.labels)}: {r.dp:+.0%} for Hidden King"
        if r.team == "Team0"
        else f"{_clock(r.start)}  {'; '.join(r.labels)}: {-r.dp:+.0%} for Archmother"
        for r in out.itertuples()
    ]
    return out.reset_index(drop=True)


def describe(moment: pd.Series) -> str:
    """One line for the page: what happened, who it helped, and why (per the model)."""
    why = f" (mostly {', '.join(moment.causes)})" if moment.causes else ""
    return f"{moment.text}{why}"
