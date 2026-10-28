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
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
import yaml

from deadlock_predictor.app.helpers import (
    api_error_message,
    history,
    live_states,
    live_table,
    pick_game,
    plot_history,
)
from deadlock_predictor.data.live import fetch_active
from deadlock_predictor.live.collect import load
from deadlock_predictor.models.refit import refit_path
from deadlock_predictor.viz.curve import CONFIG_PATH, load_models
from deadlock_predictor.viz.explorer import (
    Match,
    fight_table,
    load_match,
    map_figure,
    match_figure,
    player_table,
)

REFRESH_S = 60


@st.cache_resource
def deploy_models():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    return load_models(config), joblib.load(refit_path(config["gbm_live"], "deploy"))


@st.cache_data(show_spinner=False, max_entries=20)
def explore(match_id: int, _models) -> Match:
    return load_match(match_id, _models)


def lookup_page(models) -> None:
    st.header("Match lookup")
    st.caption(
        "Any ranked 6v6 match, fetched from deadlock-api.com. The curve has a point at every "
        "stat snapshot (filled) and at every building, mid-boss and teamfight (open); between "
        "snapshots, souls and other stats are interpolated. Hover anything for details, drag "
        "to zoom, double-click to reset."
    )
    match_id = st.text_input("Match id", placeholder="e.g. 124832378").strip()
    if not match_id:
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
    if m.failed:
        st.warning(f"This match fails training filters ({', '.join(m.failed)}): read it with care.")
    st.plotly_chart(match_figure(m), use_container_width=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Map")
        minute = st.slider(
            "Minute", 0.0, m.duration_s / 60, min(10.0, m.duration_s / 60), step=0.25
        )
        st.plotly_chart(map_figure(m, minute * 60), use_container_width=True)
    with right:
        st.subheader("Teamfights")
        fights = fight_table(m)
        if fights.empty:
            st.info("No teamfights (3+ deaths close together).")
        else:
            st.dataframe(fights, hide_index=True)
        st.subheader("Players")
        st.dataframe(player_table(m), hide_index=True)
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
            fig = plot_history(past, int(choice))
            st.pyplot(fig)
            plt.close(fig)
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
