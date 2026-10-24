"""Build the snapshot table: download → filter → features → split.

Run: python -m deadlock_predictor.data.pipeline [--force] [--skip-extract]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from deadlock_predictor.data.dumps import connect, fetch_manifest
from deadlock_predictor.data.extract import extract
from deadlock_predictor.data.split import split_boundaries, split_case
from deadlock_predictor.features.snapshots import build_snapshots, filter_matches

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "data.yaml"


def run(config: dict, force: bool = False, skip_extract: bool = False) -> dict:
    """Run every stage; returns the summary that is also written to disk."""
    filters, settings = config["filters"], config["pipeline"]
    start_time = datetime.fromisoformat(filters["start_time"])
    raw_dir = ROOT / settings["raw_dir"]
    out_path = ROOT / settings["snapshots_path"]
    con = connect()

    if not skip_extract:
        extract(
            con,
            fetch_manifest(config["source"]["manifest_url"]),
            raw_dir,
            start_time,
            match_mode=filters["match_mode"],
            game_mode=filters["game_mode"],
            refresh_latest=settings["refresh_latest"],
            force=force,
        )

    con.sql(f"CREATE VIEW raw AS SELECT * FROM read_parquet('{raw_dir.as_posix()}/*.parquet')")
    report = filter_matches(con)
    print("matches:", report)

    fractions = settings["split"]
    boundaries = split_boundaries(con, train=fractions["train"], validation=fractions["validation"])
    snapshots = build_snapshots(con).project(f"*, {split_case(boundaries)} AS split")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    snapshots.write_parquet(str(tmp))
    tmp.replace(out_path)

    counts = con.sql(
        f"""SELECT split, count(DISTINCT match_id), count(*)
            FROM '{out_path.as_posix()}' GROUP BY split"""
    ).fetchall()
    summary = {
        "built_at": datetime.now(UTC).isoformat(),
        "start_time": start_time.isoformat(),
        "filters": report,
        "split_fractions": fractions,
        "validation_starts": boundaries[0].isoformat(),
        "test_starts": boundaries[1].isoformat(),
        "splits": {s: {"matches": m, "rows": r} for s, m, r in sorted(counts)},
    }
    summary_path = ROOT / settings["summary_path"]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["splits"], indent=2))
    print(f"Wrote {out_path} and {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download every partition")
    parser.add_argument(
        "--skip-extract", action="store_true", help="rebuild features from data already in raw/"
    )
    args = parser.parse_args()
    run(yaml.safe_load(CONFIG_PATH.read_text()), force=args.force, skip_extract=args.skip_extract)


if __name__ == "__main__":
    main()
