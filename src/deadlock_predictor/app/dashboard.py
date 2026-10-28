"""Deadlock win-probability dashboard.

Run: streamlit run src/deadlock_predictor/app/dashboard.py

Two pages:
- Match lookup: the interactive explorer of any finished ranked match (`viz.explorer`).
- Live games: ranked games in progress with the live model's P(Hidden King wins), refreshed every
  minute; pick one to see its curve so far (from the collector's log, if it's running, and
  from the polls made while this page is open).
"""

from __future__ import annotations

import time

import joblib
import pandas as pd
import requests
import streamlit as st
import yaml

from deadlock_predictor.app import match_page
from deadlock_predictor.app.helpers import (
    api_error_message,
    history,
    live_states,
    live_table,
    pick_game,
    plot_history,
    portraits,
)
from deadlock_predictor.data.live import fetch_active
from deadlock_predictor.live.collect import load
from deadlock_predictor.models.refit import refit_path
from deadlock_predictor.viz.curve import CONFIG_PATH, load_models
from deadlock_predictor.viz.explorer import Match, load_match
from deadlock_predictor.viz.theme import TEAM_COLORS

REFRESH_S = 60


@st.cache_resource
def deploy_models():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    return load_models(config), joblib.load(refit_path(config["gbm_live"], "deploy"))


@st.cache_data(show_spinner=False, max_entries=20)
def explore(match_id: int, _models) -> Match:
    return load_match(match_id, _models)


def lookup_page(models) -> None:
    match_id = st.text_input(
        "Match id", placeholder="e.g. 123278367", label_visibility="collapsed"
    ).strip()
    if not match_id:
        st.markdown("### Look up a match")
        st.caption(
            "Any ranked 6v6 match from deadlock-api.com: scoreboard, win probability at every "
            "snapshot, building, mid-boss and fight, the key moments, every player's timeline "
            "and build, the fights, and an animated replay on the minimap."
        )
        return
    if not match_id.isdigit():
        st.error("A match id is a number.")
        return
    try:
        with st.spinner("Loading the match..."):
            m = explore(int(match_id), models)
    except (requests.RequestException, ValueError, KeyError) as error:
        st.error(api_error_message(match_id, error))
        return
    match_page.render(m)
    with st.expander("Curve numbers"):
        st.dataframe(m.points.assign(minute=m.points.t / 60).set_index("minute"))


def live_page(live_model) -> None:
    st.header("Live games")
    st.caption(
        "Ranked games in progress. The live model sees only what the API shows live "
        "(game time, souls, buildings, heroes), so it is less sure than the post-game "
        f"curves. Refreshes every {REFRESH_S} s."
    )
    st.session_state.setdefault("seen", pd.DataFrame())
    st.session_state.setdefault("shown", [])  # match ids in the order the table showed them

    def clicked():
        # Row positions refer to the table as the user saw it (`shown`), not to the next
        # poll's order. Copying the match id into the box makes both ways of choosing agree.
        rows = st.session_state.games.selection.rows
        if rows:
            st.session_state.follow = st.session_state.shown[rows[0]]

    @st.fragment(run_every=REFRESH_S)
    def board():
        try:
            states = live_states(fetch_active(), live_model, time.time())
        except requests.RequestException as error:
            st.warning(f"The API didn't answer: {error}")
            return
        if states.empty:
            st.info("No ranked games in progress right now.")
            return
        seen = pd.concat([st.session_state.seen, states], ignore_index=True)
        st.session_state.seen = seen
        table = live_table(states)
        st.session_state.shown = table.match.tolist()

        left, right = st.columns([2, 3])
        with left:
            wanted = st.text_input(
                "Match id", key="follow", placeholder="click a game, or type an id"
            ).strip()
            st.caption(f"{len(states)} games, updated {pd.Timestamp.now():%H:%M:%S}")
            st.dataframe(
                table,
                key="games",
                hide_index=True,
                height=480,
                on_select=clicked,
                selection_mode="single-row",
            )
        with right:
            choice = pick_game(st.session_state.shown, wanted)
            if choice is None:
                past = history(int(wanted), load(), seen) if wanted.isdigit() else pd.DataFrame()
                if past.empty:
                    st.warning(f"Match {wanted} isn't a ranked 6v6 game in progress right now.")
                    return
                # Seen live earlier: keep its curve so far.
                st.info(f"Match {wanted} has ended: its full curve is in Match lookup.")
                choice = wanted
            past = history(int(choice), load(), seen)
            game = states[states.match_id == int(choice)]
            if len(game):
                g = game.iloc[0]
                st.markdown(
                    f"<div style='color:{TEAM_COLORS['Team0']}'>Hidden King "
                    f"{portraits(g.heroes_team0)}</div>"
                    f"<div style='color:{TEAM_COLORS['Team1']}; margin-top:4px'>Archmother "
                    f"{portraits(g.heroes_team1)}</div>",
                    unsafe_allow_html=True,
                )
            st.plotly_chart(plot_history(past, int(choice)), use_container_width=True)
            if len(past) < 3:
                st.caption(
                    "The live API only gives the current state: the curve starts when you "
                    "start watching and gains a point every minute."
                )

    board()


def main() -> None:
    st.set_page_config(page_title="Deadlock win probability", layout="wide")
    models, live_model = deploy_models()
    page = st.sidebar.radio("Page", ["Match lookup", "Live games"])
    if page == "Match lookup":
        lookup_page(models)
    else:
        live_page(live_model)


main()
