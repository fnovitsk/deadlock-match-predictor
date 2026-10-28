"""The dashboard's logic, kept apart from Streamlit so it can be tested."""

from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

from deadlock_predictor.data.assets import hero
from deadlock_predictor.data.live import OBJ_COLUMNS, heroes_frame, live_state
from deadlock_predictor.models.data import predict_saved
from deadlock_predictor.viz.theme import TEAM_COLORS, TEAM_FILL, TEXT, styled


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


def plot_history(states: pd.DataFrame, match_id: int) -> go.Figure:
    """P(Hidden King wins) and the souls lead of one game in progress, so far (Plotly)."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06
    )
    minutes = states.t / 60
    fig.add_hrect(
        y0=0.5,
        y1=1,
        fillcolor=TEAM_FILL["Team0"],
        line_width=0,
        row=1,
        col=1,
        exclude_empty_subplots=False,
    )
    fig.add_hrect(
        y0=0,
        y1=0.5,
        fillcolor=TEAM_FILL["Team1"],
        line_width=0,
        row=1,
        col=1,
        exclude_empty_subplots=False,
    )
    fig.add_trace(
        go.Scatter(
            x=minutes,
            y=states.p,
            mode="lines+markers",
            name="P(Hidden King wins)",
            line={"color": TEXT, "width": 3},
            hovertemplate="%{x:.1f} min: %{y:.0%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    lead = states.net_worth_diff
    for team, sign in (("Team0", 1), ("Team1", -1)):
        fig.add_trace(
            go.Scatter(
                x=minutes,
                y=lead.clip(lower=0) if sign > 0 else lead.clip(upper=0),
                fill="tozeroy",
                mode="lines",
                line={"color": TEAM_COLORS[team]},
                name=f"souls lead ({'Hidden King' if sign > 0 else 'Archmother'})",
                hovertemplate="%{x:.1f} min: %{y:+,.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text="P(Hidden King)", row=1, col=1)
    fig.update_yaxes(title_text="souls lead", row=2, col=1)
    fig.update_xaxes(title_text="minute (game clock)", row=2, col=1)
    fig.update_layout(title=f"Match {match_id} (live)", showlegend=False)
    return styled(fig, 520)


def portraits(hero_ids) -> str:
    """HTML: a row of hero portraits (names on hover)."""
    out = []
    for h in hero_ids:
        info = hero(int(h))
        if info.portrait:
            out.append(
                f'<img src="{info.portrait}" title="{info.name}" '
                'style="width:40px;height:40px;border-radius:6px;margin-right:4px"/>'
            )
        else:
            out.append(f"<span style='margin-right:8px'>{info.name}</span>")
    return "".join(out)


def api_error_message(match_id: str, error: Exception) -> str:
    """A readable reason the API couldn't give a match.

    deadlock-api serves matches it has already processed at once. Newer ones it fetches from
    Steam, which can be busy (503) and is rate-limited per IP (429).
    """
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status == 429:
        try:
            wait = response.json()["error"]["next_request_in"]
            when = f" Try again in about {max(1, round(wait / 60))} min."
        except (ValueError, KeyError, TypeError):
            when = ""
        return (
            f"Match {match_id} isn't processed by deadlock-api yet, and fetching it from "
            f"Steam is rate-limited (a few per hour).{when}"
        )
    if status == 503:
        return (
            f"Match {match_id} isn't processed by deadlock-api yet, and Steam is busy. "
            "Try again in a few minutes."
        )
    if status == 404:
        return f"No match {match_id} on deadlock-api."
    if isinstance(error, requests.RequestException):
        return f"Couldn't reach deadlock-api for match {match_id}: {error}"
    return f"Couldn't load match {match_id}: {error}"
