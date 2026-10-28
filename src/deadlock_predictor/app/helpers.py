"""The dashboard's logic, kept apart from Streamlit so it can be tested."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import pandas as pd

from deadlock_predictor.data.live import OBJ_COLUMNS, heroes_frame, live_state
from deadlock_predictor.models.data import predict_saved


def live_states(games: list[dict], model: dict, now: float | None = None) -> pd.DataFrame:
    """One row per game in progress, with the live model's P(Team0 wins) as `p`."""
    now = time.time() if now is None else now
    if not games:
        return pd.DataFrame()
    states = pd.DataFrame([live_state(g, now) for g in games])
    states["p"] = predict_saved(model, states, heroes_frame(states))
    return states


def live_table(states: pd.DataFrame) -> pd.DataFrame:
    """What the live page lists: game time, leads and P(Team0 wins), longest games first."""
    if states.empty:
        return pd.DataFrame()
    table = pd.DataFrame(
        {
            "match": states.match_id.astype(str),
            "minute": (states.t // 60).astype(int),
            "souls lead (Hidden King)": states.net_worth_diff.astype(int),
            "buildings lead (Hidden King)": states[OBJ_COLUMNS].sum(axis=1).astype(int),
            "P(Hidden King wins)": states.p.round(3),
        }
    )
    return table.sort_values("minute", ascending=False, ignore_index=True)


def pick_game(live: list[str], wanted: str) -> str | None:
    """The game to plot: `wanted` if it is live, the first (longest) game if nothing is
    wanted, else None."""
    if not wanted:
        return live[0] if live else None
    return wanted if wanted in live else None


def history(match_id: int, *frames: pd.DataFrame) -> pd.DataFrame:
    """Every logged state of one game across `frames` (collector log, this session),
    one per poll, in time order."""
    parts = [f[f.match_id == match_id] for f in frames if not f.empty]
    if not parts:
        return pd.DataFrame()
    states = pd.concat(parts, ignore_index=True).drop_duplicates("polled_at", keep="last")
    return states.sort_values("t", ignore_index=True)


def plot_history(states: pd.DataFrame, match_id: int):
    """P(Team0 wins) and the souls lead of one game in progress, so far."""
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, height_ratios=[3, 2])
    minutes = states.t / 60
    ax.axhspan(0.5, 1, color="tab:orange", alpha=0.06)
    ax.axhspan(0, 0.5, color="tab:blue", alpha=0.06)
    ax.axhline(0.5, color="grey", lw=0.8)
    ax.plot(minutes, states.p, marker=".", color="black")
    ax.set(ylim=(0, 1), ylabel="P(Hidden King wins)", title=f"Match {match_id} (live)")
    bx.axhline(0, color="grey", lw=0.8)
    bx.plot(minutes, states.net_worth_diff, color="tab:green")
    bx.set(xlabel="minute (since the match started)", ylabel="souls lead (Hidden King)")
    fig.tight_layout()
    return fig
