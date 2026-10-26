"""Deadlock win-probability dashboard.

Run: streamlit run src/deadlock_predictor/app/dashboard.py

Two pages:
- Match lookup: the win-probability curve of any finished ranked match (`viz.curve`).
- Live games: ranked games in progress with the live model's P(Team0 wins), refreshed every
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

from deadlock_predictor.app.helpers import history, live_states, live_table, plot_history
from deadlock_predictor.data.live import fetch_active
from deadlock_predictor.live.collect import load
from deadlock_predictor.models.refit import refit_path
from deadlock_predictor.viz.curve import CONFIG_PATH, draw, load_models

REFRESH_S = 60


@st.cache_resource
def deploy_models():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    return load_models(config), joblib.load(refit_path(config["gbm_live"], "deploy"))


def lookup_page(models) -> None:
    st.header("Match lookup")
    st.caption(
        "Any ranked 6v6 match: from the local dump, or fetched from deadlock-api.com. "
        "One point per stat snapshot (every 3-5 minutes)."
    )
    match_id = st.text_input("Match id", placeholder="e.g. 124832378")
    if not match_id:
        return
    if not match_id.isdigit():
        st.error("A match id is a number.")
        return
    try:
        with st.spinner("Loading the match..."):
            fig, curve, events = draw(int(match_id), models=models)
    except (requests.RequestException, ValueError) as error:
        st.error(f"Couldn't draw match {match_id}: {error}")
        return
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("Numbers"):
        st.dataframe(curve.assign(minute=curve.t / 60).set_index("minute").drop(columns="t"))
        st.dataframe(events)


def live_page(live_model) -> None:
    st.header("Live games")
    st.caption(
        "Ranked games in progress. The live model sees only what the API shows live "
        "(game time, souls, buildings, heroes), so it is less sure than the post-game "
        f"curves. Refreshes every {REFRESH_S} s."
    )
    if "seen" not in st.session_state:
        st.session_state.seen = pd.DataFrame()

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
        st.caption(f"{len(states)} games, updated {pd.Timestamp.now():%H:%M:%S}")
        table = live_table(states)
        choice = st.selectbox("Game", table.match, format_func=lambda m: f"match {m}")
        st.dataframe(table, hide_index=True, height=250)
        if choice:
            past = history(int(choice), load(), seen)
            fig = plot_history(past, int(choice))
            st.pyplot(fig)
            plt.close(fig)

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
