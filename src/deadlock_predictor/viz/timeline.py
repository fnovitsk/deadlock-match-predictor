"""Win probability at any second of a finished match, not just at stat snapshots.

Training rows exist only at stat snapshots (minutes 3, 6, 9, 12, 15, then every 5). To make
the curve move when things happen, `features_at` builds the same model inputs at any
time t:
- buildings and mid-boss: exact, counting events with 0 < time ≤ t, as in training;
- kills: the snapshot's kills plus the kills since it, from the death list, so they change
  at the second of each kill and equal the training value at every snapshot;
- souls, levels, assists, damage, farm: linear between the surrounding snapshots. That
  uses the next snapshot, up to 5 minutes later: a best guess of the state at t, not
  something a live viewer could have known;
- rank: constant.
At snapshot times the rows equal the training rows exactly (tested).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from deadlock_predictor.features.snapshots import FEATURES, LABEL, OBJECTIVE_GROUPS, STATS
from deadlock_predictor.viz.curve import EVENT_NAMES, MID_BOSS, TEAM_NAMES, match_curve

SIGN = {"Team0": 1, "Team1": -1}
INTERPOLATED = [s for s in STATS if s != "kills"]


def _group(kind: str) -> str | None:
    # TitanShieldGenerator before Titan (shared prefix), as in features.snapshots.
    for prefix, group in sorted(OBJECTIVE_GROUPS.items(), key=lambda kv: -len(kv[0])):
        if kind.startswith(prefix):
            return group
    return None


def stat_knots(rows: pd.DataFrame) -> pd.DataFrame:
    """Team0 − Team1 sums of every stat at each stat time stamp (the final one included),
    where all 12 players have it, starting from 0 at t = 0."""
    long = rows[["team", "stats.time_stamp_s", *(f"stats.{s}" for s in STATS)]].explode(
        ["stats.time_stamp_s", *(f"stats.{s}" for s in STATS)]
    )
    long.columns = ["team", "t", *STATS]
    long = long.astype({"t": float, **dict.fromkeys(STATS, float)})
    sign = long.team.map(SIGN).to_numpy()[:, None]
    long[STATS] = long[STATS].to_numpy() * sign
    counts = long.groupby("t").size()
    knots = long.groupby("t")[STATS].sum()[counts == 12]
    knots = knots.add_suffix("_diff").reset_index()
    zero = pd.DataFrame({"t": [0.0], **{f"{s}_diff": [0.0] for s in STATS}})
    return pd.concat([zero, knots[knots.t > 0]], ignore_index=True)


def objective_events(rows: pd.DataFrame) -> pd.DataFrame:
    """Buildings destroyed and mid-boss claims/kills: t, kind, group, team that scored,
    and for buildings the owner and the data's own name (e.g. 'Tier1Lane3')."""
    first = rows.iloc[0]
    other = {"Team0": "Team1", "Team1": "Team0"}
    events = []
    for kind, owner, t in zip(
        first["objectives.team_objective"],
        first["objectives.team"],
        first["objectives.destroyed_time_s"],
        strict=True,
    ):
        group = _group(kind)
        if t > 0 and group:
            events.append(
                {
                    "t": float(t),
                    "kind": EVENT_NAMES[group],
                    "group": group,
                    "team": other[owner],
                    "owner": owner,
                    "name": kind,
                }
            )
    for claimed, killed, t in zip(
        first["mid_boss.team_claimed"],
        first["mid_boss.team_killed"],
        first["mid_boss.destroyed_time_s"],
        strict=True,
    ):
        if t > 0:
            events.append(
                {
                    "t": float(t),
                    "kind": MID_BOSS,
                    "group": "mid_boss",
                    "team": claimed,
                    "killed_by": killed,
                }
            )
    columns = ["t", "kind", "group", "team", "killed_by", "owner", "name"]
    return pd.DataFrame(events, columns=columns).sort_values("t", ignore_index=True)


def bounty_steps(stats: pd.DataFrame, events: pd.DataFrame, times) -> tuple[np.ndarray, np.ndarray]:
    """Team0 − Team1 souls from objectives, as steps at the events that earned them.

    `stats` is `data.match_detail.player_stats` (its `gold_boss` is each player's souls
    from guardians, walkers and the mid-boss). Each snapshot interval's rise in a team's
    objective souls is split equally over that team's buildings and mid-boss kills in the
    interval; a rise with no event stays out (it is interpolated with the other souls).
    Returns the steps at each time in `times` and at each stat time stamp (`stats.t`).
    """
    gold = stats.groupby(["t", "team"]).gold_boss.sum().unstack(fill_value=0)
    gold = gold.reindex(columns=["Team0", "Team1"], fill_value=0)
    knots_t = np.concatenate([[0.0], gold.index.to_numpy(float)])
    rises = gold.diff().fillna(gold)
    scorer = events.killed_by.where(events.group == "mid_boss", events.team)
    steps = []  # (t, signed souls)
    for (lo, hi), (_, rise) in zip(
        zip(knots_t[:-1], knots_t[1:], strict=True), rises.iterrows(), strict=True
    ):
        for team in ("Team0", "Team1"):
            mine = events[(events.t > lo) & (events.t <= hi) & (scorer == team)]
            for t in mine.t:
                steps.append((t, SIGN[team] * rise[team] / len(mine)))
    step_t = np.array([t for t, _ in steps], dtype=float)
    step_v = np.array([v for _, v in steps], dtype=float)

    def at(ts):
        return np.array([step_v[step_t <= t].sum() for t in ts])

    return at(np.asarray(times, dtype=float)), at(knots_t[1:])


def features_at(
    snaps: pd.DataFrame,
    rows: pd.DataFrame,
    deaths: pd.DataFrame,
    times,
    stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Model input rows (as `features.snapshots`) at each time in `times`.

    `snaps` are the match's training snapshot rows (`viz.curve.match_features`), `rows`
    its dump rows, `deaths` its death list (`data.match_detail.deaths`). With `stats`
    (`data.match_detail.player_stats`), the souls a building or mid-boss gives jump at the
    second it falls (`bounty_steps`) instead of being spread between snapshots.
    """
    times = np.sort(np.asarray(times, dtype=float))
    snaps = snaps.sort_values("t")
    snap_t = snaps.t.astype(float).to_numpy()
    out = pd.DataFrame({"match_id": snaps.match_id.iloc[0], "t": times})

    knots = stat_knots(rows)
    for s in INTERPOLATED:
        out[f"{s}_diff"] = np.interp(times, knots.t, knots[f"{s}_diff"])
    events = objective_events(rows)
    if stats is not None:
        steps, steps_at_stamps = bounty_steps(stats, events, times)
        stamps = pd.Series(steps_at_stamps, index=np.sort(stats.t.unique()))
        knot_steps = stamps.reindex(knots.t).fillna(0).to_numpy()  # 0 at t = 0
        rest = knots.net_worth_diff.to_numpy() - knot_steps
        out["net_worth_diff"] = np.interp(times, knots.t, rest) + steps

    # Kills: the last snapshot's value plus the enemy kills since then.
    credited = deaths[deaths.killer_team.notna()]
    kill_t = credited.t.to_numpy()
    kill_sign = credited.killer_team.map(SIGN).to_numpy()

    def kills_by(t):
        return float(kill_sign[kill_t <= t].sum())

    base_t = np.concatenate([[0.0], snap_t])
    base_k = np.concatenate([[0.0], snaps.kills_diff.astype(float).to_numpy()])
    last = np.searchsorted(base_t, times, side="right") - 1
    out["kills_diff"] = [
        base_k[i] + kills_by(t) - kills_by(base_t[i]) for i, t in zip(last, times, strict=True)
    ]

    def signed_counts(e: pd.DataFrame, team_column: str) -> list[float]:
        """For each time: events so far by Team0 minus events so far by Team1."""
        signs = e[team_column].map(SIGN).astype(float).fillna(0).to_numpy()
        return [float(signs[e.t.to_numpy() <= t].sum()) for t in times]

    for g in OBJECTIVE_GROUPS.values():
        out[f"obj_{g}_diff"] = signed_counts(events[events.group == g], "team")
    boss = events[events.group == "mid_boss"]
    out["mid_boss_claims_diff"] = signed_counts(boss, "team")
    out["mid_boss_kills_diff"] = signed_counts(boss, "killed_by")
    out["badge_mean"] = snaps.badge_mean.iloc[0]
    out[LABEL] = snaps[LABEL].iloc[0]
    return out[["match_id", *FEATURES, LABEL]]  # FEATURES starts with t


def event_points(
    snaps, heroes, rows, deaths, stats, models, duration_s, extra: pd.DataFrame | None = None
) -> pd.DataFrame:
    """The curve at every snapshot and event: t, kind ('snapshot', 'event', or 'before',
    one second before an event so the jump shows), what triggered it, the models'
    P(Team0 wins) and the final model's grouped contributions.

    `extra` adds events of your own (t, label), e.g. the ends of teamfights.
    """
    first_t = float(snaps.t.min())
    triggers = [{"t": float(t), "kind": "snapshot", "trigger": "stat snapshot"} for t in snaps.t]
    events = objective_events(rows)
    labels = [(e.t, f"{e.kind} ({TEAM_NAMES[e.team]})") for e in events.itertuples()]
    if extra is not None:
        labels += list(zip(extra.t, extra.label, strict=True))
    for t, label in labels:
        if first_t <= t < duration_s:
            triggers.append({"t": float(t), "kind": "event", "trigger": label})
            triggers.append({"t": float(t) - 1, "kind": "before", "trigger": ""})
    points = pd.DataFrame(triggers)
    # One row per second: an event at a snapshot time is kept as the event.
    order = {"event": 0, "snapshot": 1, "before": 2}
    points = (
        points.sort_values(["t", "kind"], key=lambda c: c.map(order) if c.name == "kind" else c)
        .groupby("t", as_index=False)
        .agg({"kind": "first", "trigger": lambda s: "; ".join(x for x in s if x)})
    )
    points = points[points.t >= first_t].reset_index(drop=True)
    rows_at = features_at(snaps, rows, deaths, points.t, stats)
    curve = match_curve(rows_at, heroes, models)
    return pd.concat([points, curve.drop(columns="t")], axis=1)
