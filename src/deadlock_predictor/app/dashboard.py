"""Deadlock win-probability dashboard.

Run: streamlit run src/deadlock_predictor/app/dashboard.py

One page: the interactive explorer of any finished ranked match (`viz.explorer`). A live
games page was removed: the API can't give a game's history from minute 0 (see
`docs/learning-notes.md`). The live model and its tooling (`data.live`, `live.collect`,
`live.score`) stay for later.
"""

from __future__ import annotations

import requests
import streamlit as st
import yaml

from deadlock_predictor.app import match_page
from deadlock_predictor.app.helpers import api_error_message
from deadlock_predictor.viz.curve import CONFIG_PATH, load_models
from deadlock_predictor.viz.explorer import Match, load_match


@st.cache_resource
def deploy_models():
    return load_models(yaml.safe_load(CONFIG_PATH.read_text()))


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
            "Any ranked 6v6 match from deadlock-api.com: the scoreboard, the win probability at "
            "every snapshot, building, mid-boss and fight, the key moments, and each player's "
            "souls, level, damage and healing."
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


def main() -> None:
    st.set_page_config(page_title="Deadlock win probability", layout="wide")
    lookup_page(deploy_models())


main()
