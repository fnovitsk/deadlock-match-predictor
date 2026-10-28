"""Score logged live predictions against how the games ended.

Run: python -m deadlock_predictor.live.score

For every game `live.collect` logged, fetch the finished match from the API (games still
in progress, or not yet processed, are skipped and picked up on a later run). Matches that
fail a training filter (abandons, no winner) are left out, as in training. Then:
- score the live model's prediction at every logged state;
- check the live inputs: does `t = now − start_time` line up with game time? The live team
  net worth is compared with the post-game per-snapshot net worth at t + offset, for a
  range of offsets; the best offset says how far the two clocks are apart.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import yaml

from deadlock_predictor.data.live import game_time, heroes_frame
from deadlock_predictor.data.match import api_rows, fetch_match
from deadlock_predictor.features.snapshots import LABEL
from deadlock_predictor.live.collect import LIVE_DIR, load
from deadlock_predictor.models.data import predict_saved
from deadlock_predictor.models.evaluate import reliability, scores, scores_by_time
from deadlock_predictor.models.refit import refit_path
from deadlock_predictor.viz.curve import match_features

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "model.yaml"
MATCHES_PATH = LIVE_DIR / "matches.parquet"  # finished matches, as dump rows
OFFSETS = range(-300, 301, 15)  # seconds tried in the clock check


def fetch_finished(match_ids, cached: pd.DataFrame | None = None, pause: float = 0.2):
    """Dump rows of the matches that have finished, using and extending a cache."""
    have = set() if cached is None else set(cached.match_id)
    new = []
    for mid in sorted(set(match_ids) - have):
        try:
            new.append(api_rows(fetch_match(mid)))
        except (requests.RequestException, KeyError):
            continue  # still running, or not processed yet
        time.sleep(pause)
    parts = ([] if cached is None else [cached]) + new
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def outcomes(rows: pd.DataFrame) -> pd.DataFrame:
    """Per finished match: the label, the failed training filters, and the duration."""
    out = []
    for mid, match in rows.groupby("match_id"):
        _, _, failed = match_features(match)
        out.append(
            {
                "match_id": mid,
                LABEL: match.winning_team.iloc[0] == "Team0",
                "failed": failed,
                "duration_s": float(match.duration_s.iloc[0]),
            }
        )
    return pd.DataFrame(out)


def team_net_worth(rows: pd.DataFrame) -> pd.DataFrame:
    """Post-game team net worth at each snapshot: match_id, team, t, net_worth."""
    long = rows[["match_id", "team", "stats.time_stamp_s", "stats.net_worth"]].explode(
        ["stats.time_stamp_s", "stats.net_worth"]
    )
    long = long.rename(columns={"stats.time_stamp_s": "t", "stats.net_worth": "net_worth"})
    long = long.astype({"t": float, "net_worth": float})
    return long.groupby(["match_id", "team", "t"], as_index=False).net_worth.sum()


def clock_check(states: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    """Median relative gap between live and post-game team net worth, per clock offset.

    Post-game net worth is interpolated between snapshots at t_live + offset. The offset
    with the smallest gap is how much later the game clock is than `now − start_time`.
    """
    gaps = {}
    for offset in OFFSETS:
        errors = []
        for (mid, team), g in post.groupby(["match_id", "team"]):
            live = states[states.match_id == mid]
            if live.empty:
                continue
            column = "net_worth_team0" if team == "Team0" else "net_worth_team1"
            t = live.t.to_numpy() + offset
            inside = (t >= g.t.min()) & (t <= g.t.max())
            if not inside.any():
                continue
            expected = np.interp(t[inside], g.t.to_numpy(), g.net_worth.to_numpy())
            errors.append(np.abs(live[column].to_numpy()[inside] / expected - 1))
        if errors:
            gaps[offset] = float(np.median(np.concatenate(errors)))
    return pd.Series(gaps, name="median relative gap").rename_axis("offset_s").to_frame()


def run(config: dict, out_dir: Path = LIVE_DIR) -> dict:
    states = load(out_dir)
    if states.empty:
        raise SystemExit("No live states logged: run `python -m deadlock_predictor.live.collect`")
    # Recompute the game clock from the raw times, so logs written before PREGAME_S existed
    # are scored the same way as new ones.
    elapsed = (states.polled_at - states.start_time).dt.total_seconds()
    states["t"] = [game_time(e, 0) for e in elapsed]
    cached = pd.read_parquet(MATCHES_PATH) if MATCHES_PATH.exists() else None
    rows = fetch_finished(states.match_id.unique(), cached)
    if rows.empty:
        raise SystemExit("None of the logged games has finished yet")
    tmp = MATCHES_PATH.with_suffix(".tmp")
    rows.to_parquet(tmp, index=False)
    tmp.replace(MATCHES_PATH)

    games = outcomes(rows)
    clean = games[games.failed.str.len() == 0]
    scored = states.merge(clean[["match_id", LABEL, "duration_s"]], on="match_id")
    scored = scored[(scored.abandoned == 0) & (scored.t < scored.duration_s)]
    model = joblib.load(refit_path(config["gbm_live"], "deploy"))
    scored["p"] = predict_saved(model, scored, heroes_frame(scored))
    y = scored[LABEL].to_numpy()
    clock = clock_check(scored, team_net_worth(rows[rows.match_id.isin(clean.match_id)]))
    report = {
        "logged_games": int(states.match_id.nunique()),
        "finished_games": int(len(games)),
        "scored_games": int(scored.match_id.nunique()),
        "scored_states": int(len(scored)),
        "left_out": {
            "unfinished": int(states.match_id.nunique() - len(games)),
            "failed_filters": int(len(games) - len(clean)),
        },
        "live": scores(y, scored.p.to_numpy()),
        "by_minute": scores_by_time(
            (scored.t // 300 * 5).to_numpy(), y, scored.p.to_numpy(), min_rows=50
        )
        .reset_index()
        .rename(columns={"t": "minute"})
        .to_dict("records"),
        "reliability": reliability(y, scored.p.to_numpy())
        .reset_index(names="bin")
        .to_dict("records"),
        "clock": {
            "best_offset_s": int(clock.iloc[:, 0].idxmin()),
            "gap_at_best": float(clock.iloc[:, 0].min()),
            "gap_at_zero": float(clock.loc[0].iloc[0]),
        },
    }
    scored.to_parquet(out_dir / "scored.parquet", index=False)
    (out_dir / "scores.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    report = run(yaml.safe_load(CONFIG_PATH.read_text()))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("by_minute", "reliability")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
