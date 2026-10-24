# CLAUDE.md

Notes for Claude Code working in this repo.

## Project
Deadlock win-probability predictor. Given the game state at time t (net worth, kills, objectives, heroes, game time), predict P(team 0 wins). V1 produces historical win-probability curves; exploration happens in Jupyter notebooks.

Data: deadlock-api.com Parquet dumps (manifest: https://data.deadlock-api.com/v1/manifest.json), queried with DuckDB via `deadlock_predictor.data.dumps`. Scope: ranked 6v6 since the last major patch (`configs/data.yaml`). Full findings: `notebooks/01_explore_data.ipynb`. CI (`.github/workflows/ci.yml`) runs ruff and pytest on PRs.

Data facts:
- The dump server and API return 403 to Python's default user agent; always use `dumps.USER_AGENT` / `dumps.connect()`.
- `match_player`: one row per player per match, 562 GB, partitioned by `match_id // 1_000_000` (higher = newer). Base file keys change when rebuilt: always build URLs from a fresh manifest. Delta files hold rows added since the last base build (including backfilled old matches), not in any base file: read base ∪ delta, de-duplicate on `(match_id, player_slot)`.
- Ranked 6v6 = `match_mode='Ranked' AND game_mode='Normal'` (~16k matches/day). Teams and `winning_team` are `'Team0'`/`'Team1'`.
- `stats.*` are per-player lists aligned with `stats.time_stamp_s`: 180, 360, ..., 900, then every 300 s, plus a final snapshot exactly at `duration_s`. **Drop the final snapshot (label leakage).**
- `objectives.*` list the owning team, type and `destroyed_time_s` (0 = never destroyed); only count `0 < destroyed_s <= t`. `objectives_mask_team*` are end-of-match: never features. `mid_boss.*` lists every kill (team_killed, team_claimed, time).
- `average_badge_team0/1` are always 0. `average_badge` is the lobby's average rank, identical on all 12 player rows: there is no per-team rank gap.
- Stats columns are unsigned (`UINTEGER`): cast to `BIGINT` before subtracting.
- The dump has few matches for Oct 20–21 2026 (~11k and ~3k vs ~22k/day): missing upstream. A later backfill would land in deltas; `--force` re-downloads.
- Team0 won 50.7% ± 0.2% in train (since the patch), ~50.0% in validation/test: a small side edge that faded.
- `game_mode_version` is constant; patch dates come from `https://api.deadlock-api.com/v1/patches/big-days`.

## The user
MCS student learning ML through this project. Explain ML concepts and the reasons behind design choices; they will ask follow-ups. Plan each milestone in plan mode and get approval before editing. Record concepts and decisions in `docs/learning-notes.md`.

Pipeline (`python -m deadlock_predictor.data.pipeline [--force] [--skip-extract]`, settings in `configs/data.yaml` → `pipeline:`):
- `data.extract` → `data/raw/match_player/partition=<N>.parquet` (skips finished partitions, refreshes the newest 2).
- `features.snapshots` filters matches (6+6 players, no abandon, scored, has winner) and builds one row per (match, t): `FEATURES`, `LABEL` (`team0_win`), all diffs Team0 − Team1.
- `data.split` adds `split` by start time (70/15/15 train/validation/test).
- Output: `data/processed/snapshots.parquet` and `summary.json`. Notebook 02 checks it. Analyse on train only.

## Commands
Conda is not on PATH in Claude's shell: use `C:\Users\fnovi\miniconda3\Scripts\conda.exe`. `gh` is not on PATH either: `C:\Program Files\GitHub CLI\gh.exe`.
`conda run` doesn't pass stdin: to pipe code, call `C:\Users\fnovi\miniconda3\envs\deadlock\python.exe -` with `PYTHONNOUSERSITE=1`.
- Run tests: `conda run -n deadlock pytest`
- Lint/format: `conda run -n deadlock ruff check .` / `ruff format .`
- All hooks: `conda run -n deadlock pre-commit run --all-files`
- Env update: `conda env update -n deadlock -f environment.yml --prune --solver=libmamba` (the classic solver hangs)

Environment gotchas:
- The user's system Python 3.11 has `--user` packages that shadow the env's; `PYTHONNOUSERSITE=1` is set in `environment.yml` and the `deadlock` kernel spec. Keep it.
- Windows PowerShell 5.1 can't open paths over 260 chars (the scratchpad path is longer); pipe code to `python -` instead.
- numpy uses OpenBLAS (pinned in `environment.yml`): MKL crashes Jupyter kernels, which start without conda activation.
- Run nbconvert as `python -m nbconvert` from the env; `jupyter.exe` dispatches to the system Python's subcommands.
- Execute notebooks into `data/interim/` (gitignored) to inspect outputs: `python -m nbconvert --to notebook --execute <nb> --ExecutePreprocessor.kernel_name=deadlock --output-dir data/interim`.

## Conventions
- Reusable logic lives in `src/deadlock_predictor/`; notebooks import from it and stay exploratory.
- Never commit data, models, or secrets (`data/`, `models/`, `*.parquet`, `.env` are gitignored).
- Notebook outputs are stripped on commit by nbstripout.
- Avoid leakage: split train/test **by match** (never by row) and **by date** (train older, test newer). Add tests that enforce this.
- Always compare new models against the baseline using log loss, Brier score, and calibration.

## Git workflow
- `main` always works. One branch per milestone (`feat/<name>`), merged by GitHub PR via `gh`.
- Small, focused commits with imperative messages ("Add snapshot extractor").
- Confirm with the user before pushing or merging.
