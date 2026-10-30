"""The match page: scoreboard, win probability with key moments, and each player's souls,
level, damage or healing. One time range (session state) zooms every chart."""

from __future__ import annotations

import html

import streamlit as st

from deadlock_predictor.data.assets import rank
from deadlock_predictor.viz import charts
from deadlock_predictor.viz.curve import TEAM_NAMES
from deadlock_predictor.viz.explorer import Match, clock, player_table
from deadlock_predictor.viz.moments import describe, key_moments
from deadlock_predictor.viz.theme import MUTED, PANEL, TEAM_COLORS, TEXT

CSS = f"""
<style>
.dl-banner {{display:flex; align-items:center; gap:18px; margin:4px 0 14px 0;}}
.dl-banner .result {{font-size:1.7rem; font-weight:700;}}
.dl-banner .meta {{color:{MUTED}; font-size:0.95rem;}}
.dl-team {{background:{PANEL}; border-radius:10px; padding:10px 12px;
  border-top:4px solid var(--team);}}
.dl-team h4 {{margin:0 0 6px 0; color:var(--team); display:flex; justify-content:space-between;}}
.dl-team h4 span {{color:{MUTED}; font-weight:400; font-size:0.9rem;}}
.dl-row {{display:grid; grid-template-columns:46px 1.3fr 0.9fr 0.8fr 0.5fr 0.8fr 0.8fr;
  align-items:center; gap:8px; padding:4px 0; border-bottom:1px solid #222834; font-size:0.92rem;}}
.dl-row.head {{color:{MUTED}; font-size:0.78rem; text-transform:uppercase; border-bottom:none;
  padding-bottom:0;}}
.dl-row img {{width:42px; height:42px; border-radius:6px; background:#0d0f14;}}
.dl-name {{font-weight:600; color:{TEXT};}}
.dl-bar {{height:4px; background:#262c38; border-radius:2px; margin-top:4px;}}
.dl-bar div {{height:4px; border-radius:2px; background:var(--team);}}
.dl-num {{text-align:right; font-variant-numeric:tabular-nums;}}
.dl-moment {{padding:6px 0; border-bottom:1px solid #222834; font-size:0.9rem;}}
.dl-moment b {{color:var(--team);}}
</style>
"""


def _img(src, title: str = "") -> str:
    if not isinstance(src, str) or not src:
        return ""
    return f'<img src="{src}" title="{html.escape(title)}"/>'


def _short(n: float) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else f"{n:.0f}"


def scoreboard(m: Match) -> str:
    """The banner and both teams' player cards, as HTML."""
    table = player_table(m).rename(columns={"hero damage": "damage"})
    table["portrait"] = m.players.portrait.to_numpy()
    table["team_key"] = m.players.team.to_numpy()
    best = max(table.souls.max(), 1)
    rank_name = rank(m.badge) if m.badge else ""
    meta = f"{clock(m.duration_s)} · match {m.match_id}" + (
        f" · {rank_name} lobby" if rank_name else ""
    )
    winner = m.winner
    out = [
        CSS,
        f'<div class="dl-banner"><div class="result" style="color:{TEAM_COLORS[winner]}">'
        f"{TEAM_NAMES[winner]} victory</div>"
        f'<div class="meta">{meta}</div></div>',
    ]
    cards = []
    for team in TEAM_COLORS:
        mine = table[table.team_key == team]
        k, d, a = mine.kills.sum(), mine.deaths.sum(), mine.assists.sum()
        rows = [
            '<div class="dl-row head"><div></div><div>hero</div><div class="dl-num">K / D / A</div>'
            '<div class="dl-num">souls</div><div class="dl-num">lvl</div>'
            '<div class="dl-num">damage</div><div class="dl-num">healing</div></div>'
        ]
        for p in mine.itertuples():
            rows.append(
                f'<div class="dl-row">{_img(p.portrait, p.hero)}'
                f'<div><div class="dl-name">{html.escape(p.hero)}</div>'
                f'<div class="dl-bar"><div style="width:{100 * p.souls / best:.0f}%">'
                "</div></div></div>"
                f'<div class="dl-num">{p.kills} / {p.deaths} / {p.assists}</div>'
                f'<div class="dl-num">{_short(p.souls)}</div>'
                f'<div class="dl-num">{p.level:.0f}</div>'
                f'<div class="dl-num">{_short(p.damage)}</div>'
                f'<div class="dl-num">{_short(p.healing)}</div></div>'
            )
        result = "WIN" if team == winner else "LOSS"
        cards.append(
            f'<div class="dl-team" style="--team:{TEAM_COLORS[team]}">'
            f"<h4>{TEAM_NAMES[team]} · {result}<span>{k} / {d} / {a} · "
            f"{_short(mine.souls.sum())} souls</span></h4>{''.join(rows)}</div>"
        )
    out.append(
        '<div style="display:grid; grid-template-columns:1fr 1fr; gap:14px;">'
        + "".join(cards)
        + "</div>"
    )
    return "".join(out)


def _range_key(m: Match) -> str:
    return f"range_{m.match_id}"


def _zoom(key: str, start: float, end: float, total: float) -> None:
    st.session_state[key] = (max(0.0, start / 60 - 1.0), min(total, end / 60 + 1.0))


def render(m: Match) -> None:
    st.markdown(scoreboard(m), unsafe_allow_html=True)
    if m.failed:
        st.warning(f"This match fails training filters ({', '.join(m.failed)}): read it with care.")

    total = round(m.duration_s / 60 + 0.5, 1)
    key = _range_key(m)
    if key not in st.session_state:
        st.session_state[key] = (0.0, total)
    xrange = st.slider("Time range (minutes): zooms every chart", 0.0, total, step=0.5, key=key)

    curve, side = st.columns([3, 1])
    with curve:
        st.plotly_chart(charts.win_probability(m, xrange), width="stretch")
        st.plotly_chart(charts.moves(m, xrange), width="stretch")
    with side:
        st.markdown("#### Key moments")
        st.caption("The biggest swings in win probability, and what drove them (per the model).")
        for i, mo in key_moments(m.points, m.fights).iterrows():
            st.markdown(
                f'<div class="dl-moment" style="--team:{TEAM_COLORS[mo.team]}">'
                f"<b>{mo.dp:+.0%}</b> {html.escape(describe(mo))}</div>",
                unsafe_allow_html=True,
            )
            st.button(
                "Zoom",
                key=f"zoom_{m.match_id}_{i}",
                on_click=_zoom,
                args=(key, mo.start, mo.end, total),
            )

    st.markdown("#### Players")
    stat = st.segmented_control(
        "Per player",
        ["souls", "level", "hero damage", "healing"],
        default="souls",
        key=f"stat_{m.match_id}",
    )
    column = {
        "souls": "net_worth",
        "level": "level",
        "hero damage": "player_damage",
        "healing": "player_healing",
    }[stat or "souls"]
    st.plotly_chart(charts.souls(m, column, xrange), width="stretch")
    st.plotly_chart(charts.souls_lead(m, xrange), width="stretch")
