"""Win-probability curve for one match.

Run: python -m deadlock_predictor.viz.curve <match_id> [--out PATH]

The match comes from the local dump or, if it isn't there, from the API
(`data.match.match_rows`). The same feature code as in training turns it into one row per
stat snapshot, and the deploy models (`models.refit --all`) predict P(Team0 wins) at each.
Snapshots exist only every 3-5 minutes, so the curve has one point per snapshot; events in
between (towers, mid-boss) are marked on the time axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from deadlock_predictor.data.match import match_rows, register
from deadlock_predictor.features.heroes import build_heroes
from deadlock_predictor.features.snapshots import OBJECTIVE_GROUPS, build_snapshots, filter_matches
from deadlock_predictor.models.data import model_inputs
from deadlock_predictor.models.refit import cells, refit_path

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "model.yaml"
OUT_DIR = ROOT / "reports" / "curves"
TEAM_COLORS = {"Team0": "tab:orange", "Team1": "tab:blue"}

# Feature groups for "what moved the probability" (SHAP contributions summed per group).
GROUPS = {
    "net worth": ["net_worth_diff"],
    "fights": ["kills_diff", "assists_diff", "player_damage_diff"],
    "levels & farm": ["level_diff", "creep_kills_diff", "denies_diff"],
    "objectives": [f"obj_{g}_diff" for g in OBJECTIVE_GROUPS.values()] + ["boss_damage_diff"],
    "mid-boss": ["mid_boss_claims_diff", "mid_boss_kills_diff"],
    "heroes": [],  # every hero_<id>_diff column
    "time & rank": ["t", "badge_mean"],
}
SHORT = {"tier1": "T1", "tier2": "T2", "barracks": "B", "shield_gen": "SG", "titan": "Ti"}


def load_models(config: dict | None = None) -> dict[str, dict]:
    """The deploy models: {"baseline": entry, "final": entry}."""
    config = config or yaml.safe_load(CONFIG_PATH.read_text())
    names = cells(config)  # ["baseline", final choice]; one name if they are the same
    paths = {"baseline": names[0], "final": names[-1]}
    return {k: joblib.load(refit_path(config[cell], "deploy")) for k, cell in paths.items()}


def match_features(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(snapshots, hero picks, failed filters) for one match's player rows."""
    con = duckdb.connect()
    register(con, rows)
    report = filter_matches(con, keep_failing=True)
    checks = ("full_teams", "no_abandon", "scored", "has_winner")
    failed = [c for c in checks if report[c]]
    return build_snapshots(con).df(), build_heroes(con).df(), failed


def group_contributions(contrib: pd.DataFrame) -> pd.DataFrame:
    """Sum per-feature contributions into GROUPS (unlisted features go to "time & rank")."""
    out = {}
    for name, columns in GROUPS.items():
        if name == "heroes":
            columns = [c for c in contrib.columns if c.startswith("hero_")]
        out[name] = contrib[[c for c in columns if c in contrib.columns]].sum(axis=1)
    listed = {c for cols in GROUPS.values() for c in cols} | {
        c for c in contrib.columns if c.startswith("hero_")
    }
    out["time & rank"] += contrib[[c for c in contrib.columns if c not in listed]].sum(axis=1)
    return pd.DataFrame(out, index=contrib.index)


def match_curve(snaps: pd.DataFrame, heroes: pd.DataFrame, models: dict[str, dict]) -> pd.DataFrame:
    """One row per snapshot: t, P(Team0 wins) per model, and the final model's grouped
    contributions to its logit (when it can explain itself, like LightGBM)."""
    curve = pd.DataFrame({"t": snaps.t.astype(float).to_numpy()})
    for name, saved in models.items():
        X = model_inputs(saved, snaps, heroes)
        curve[f"p_{name}"] = saved["model"].predict_proba(X)[:, 1]
        if name == "final" and hasattr(saved["model"], "contributions"):
            groups = group_contributions(saved["model"].contributions(X).reset_index(drop=True))
            curve = pd.concat([curve, groups], axis=1)
    return curve


def match_events(rows: pd.DataFrame) -> pd.DataFrame:
    """Objective kills and mid-boss claims: t, kind, and the team that scored them."""
    first = rows.iloc[0]
    other = {"Team0": "Team1", "Team1": "Team0"}
    events = []
    names = first["objectives.team_objective"]
    for kind, owner, t in zip(
        names, first["objectives.team"], first["objectives.destroyed_time_s"], strict=True
    ):
        group = next(
            (
                g
                for p, g in sorted(OBJECTIVE_GROUPS.items(), key=lambda kv: -len(kv[0]))
                if kind.startswith(p)
            ),
            None,
        )
        if t > 0 and group:
            events.append({"t": float(t), "kind": SHORT[group], "team": other[owner]})
    for team, t in zip(
        first["mid_boss.team_claimed"], first["mid_boss.destroyed_time_s"], strict=True
    ):
        if t > 0:
            events.append({"t": float(t), "kind": "MB", "team": team})
    return pd.DataFrame(events, columns=["t", "kind", "team"]).sort_values("t", ignore_index=True)


def label_levels(events: pd.DataFrame, gap_min: float = 1.0) -> list[int]:
    """A row for each event label, so labels less than `gap_min` minutes apart on the same
    side don't overlap: the first row whose previous label is far enough away."""
    last: dict[str, list[float]] = {}
    levels = []
    for e in events.itertuples():
        rows = last.setdefault(e.team, [])
        minute = e.t / 60
        level = next((i for i, x in enumerate(rows) if minute - x >= gap_min), len(rows))
        if level == len(rows):
            rows.append(minute)
        rows[level] = minute
        levels.append(level)
    return levels


def plot_curve(curve, events, match_id, winner, duration_s, note="", fig=None):
    """P(Team0 wins) over time, with events and what moved the final model's prediction."""
    has_groups = "net worth" in curve.columns
    fig = fig or plt.figure(figsize=(11, 6.5 if has_groups else 4.5))
    grid = fig.add_gridspec(2 if has_groups else 1, 1, height_ratios=[3, 2][: 1 + has_groups])
    ax = fig.add_subplot(grid[0])
    minutes = curve.t / 60
    ax.axhspan(0.5, 1, color=TEAM_COLORS["Team0"], alpha=0.06)
    ax.axhspan(0, 0.5, color=TEAM_COLORS["Team1"], alpha=0.06)
    ax.axhline(0.5, color="grey", lw=0.8)
    ax.plot(minutes, curve.p_final, marker="o", color="black", label="final model")
    ax.plot(minutes, curve.p_baseline, marker=".", ls="--", color="grey", label="baseline")
    for e, level in zip(events.itertuples(), label_levels(events), strict=True):
        step = 0.05 * level
        y = 0.97 - step if e.team == "Team0" else 0.03 + step
        ax.text(
            e.t / 60,
            y,
            e.kind,
            color=TEAM_COLORS[e.team],
            fontsize=7,
            ha="center",
            va="center",
            fontweight="bold",
        )
    end = duration_s / 60
    ax.axvline(end, color=TEAM_COLORS.get(winner, "grey"), lw=2)
    ax.text(end, 0.5, f" {winner} wins", color=TEAM_COLORS.get(winner, "grey"), va="center")
    ax.set(
        ylim=(0, 1),
        xlim=(0, end + 2),
        ylabel="P(Team0 wins)",
        title=f"Match {match_id}{note}",
    )
    ax.legend(loc="lower left", fontsize=8)
    if has_groups:
        bx = fig.add_subplot(grid[1], sharex=ax)
        moves = curve[list(GROUPS)].diff().fillna(curve[list(GROUPS)])  # first: from 50%
        bottom_pos = np.zeros(len(curve))
        bottom_neg = np.zeros(len(curve))
        width = 1.2
        for i, group in enumerate(GROUPS):
            v = moves[group].to_numpy()
            pos, neg = np.clip(v, 0, None), np.clip(v, None, 0)
            color = plt.cm.tab10(i)
            bx.bar(minutes, pos, width, bottom=bottom_pos, color=color, label=group)
            bx.bar(minutes, neg, width, bottom=bottom_neg, color=color)
            bottom_pos += pos
            bottom_neg += neg
        bx.axhline(0, color="grey", lw=0.8)
        span = max(bottom_pos.max(), -bottom_neg.min(), 0.1) * 1.1
        bx.set(
            ylim=(-span, span),
            xlabel="minute",
            ylabel="Δ logit since last point\n(+ = toward Team0)",
        )
        bx.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    else:
        ax.set_xlabel("minute")
    fig.tight_layout()
    return fig


def draw(match_id: int, out: Path | None = None, models: dict | None = None):
    """Load, predict and plot one match. Returns (figure, curve, events)."""
    rows, source = match_rows(match_id)
    snaps, heroes, failed = match_features(rows)
    if snaps.empty:
        raise ValueError(f"match {match_id} has no usable snapshots")
    curve = match_curve(snaps, heroes, models or load_models())
    events = match_events(rows)
    winner = rows.winning_team.iloc[0]
    note = {"dump": " (from the local dump)", "api": " (from the API)"}[source]
    if failed:
        note += f" — fails training filters: {', '.join(failed)}"
    fig = plot_curve(curve, events, match_id, winner, float(rows.duration_s.iloc[0]), note)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
    return fig, curve, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("match_id", type=int)
    parser.add_argument("--out", type=Path, help="PNG path (default reports/curves/<id>.png)")
    args = parser.parse_args()
    out = args.out or OUT_DIR / f"{args.match_id}.png"
    _, curve, events = draw(args.match_id, out)
    table = curve.assign(minute=curve.t / 60).set_index("minute").drop(columns="t")
    print(table.round(3).to_string())
    print(f"\n{len(events)} events; saved {out}")


if __name__ == "__main__":
    main()
