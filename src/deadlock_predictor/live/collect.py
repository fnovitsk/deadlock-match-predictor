"""Log ranked games in progress every minute, to check live predictions once they end.

Run: python -m deadlock_predictor.live.collect [--minutes 120] [--every 60]

Each poll appends one row per game (`data.live.live_state`) to
`data/live/<UTC date>.parquet`. The raw API values are stored too, so later checks don't
depend on how today's code turns them into features. `live.score` scores them later.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from deadlock_predictor.data.live import fetch_active, live_state

ROOT = Path(__file__).resolve().parents[3]
LIVE_DIR = ROOT / "data" / "live"


def append(states: pd.DataFrame, path: Path) -> int:
    """Add rows to a log file, keeping one row per (match, poll). Returns the new total."""
    if path.exists():
        states = pd.concat([pd.read_parquet(path), states], ignore_index=True)
    states = states.drop_duplicates(["match_id", "polled_at"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    states.to_parquet(tmp, index=False)
    tmp.replace(path)
    return len(states)


def poll(out_dir: Path = LIVE_DIR) -> int:
    """One poll: log every ranked game in progress. Returns how many games were logged."""
    now = time.time()
    games = fetch_active()
    if not games:
        return 0
    states = pd.DataFrame([live_state(g, now) for g in games])
    day = pd.Timestamp(now, unit="s", tz="UTC").strftime("%Y-%m-%d")
    append(states, out_dir / f"{day}.parquet")
    return len(games)


def load(out_dir: Path = LIVE_DIR) -> pd.DataFrame:
    """Every logged state."""
    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=120, help="how long to collect")
    parser.add_argument("--every", type=float, default=60, help="seconds between polls")
    args = parser.parse_args()
    end = time.time() + args.minutes * 60
    while time.time() < end:
        start = time.time()
        try:
            n = poll()
            print(f"{pd.Timestamp.now(tz='UTC'):%H:%M:%S} logged {n} games", flush=True)
        except requests.RequestException as error:  # keep going through API hiccups
            print(f"poll failed: {error}", flush=True)
        time.sleep(max(0.0, args.every - (time.time() - start)))


if __name__ == "__main__":
    main()
