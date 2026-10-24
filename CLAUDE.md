# CLAUDE.md

Notes for Claude Code working in this repo.

## Project
Deadlock win-probability predictor. Given the game state at time t (net worth, kills, objectives, heroes, game time), predict P(team 0 wins). V1 produces historical win-probability curves; exploration happens in Jupyter notebooks.

Data: deadlock-api.com Parquet dumps (manifest: https://data.deadlock-api.com/v1/manifest.json), queried with DuckDB. The `match_player` table holds per-player `stats.*` snapshot arrays keyed by `stats.time_stamp_s`, plus `objectives.*`, `mid_boss.*`, `winning_team`. Scope: recent ranked 6v6 matches.

## The user
MCS student learning ML through this project. Explain ML concepts and the reasons behind design choices; they will ask follow-ups. Plan each milestone in plan mode and get approval before editing. Record concepts and decisions in `docs/learning-notes.md`.

## Commands
Conda is not on PATH in Claude's shell: use `C:\Users\fnovi\miniconda3\Scripts\conda.exe`.
- Run tests: `conda run -n deadlock pytest`
- Lint/format: `conda run -n deadlock ruff check .` / `ruff format .`
- All hooks: `conda run -n deadlock pre-commit run --all-files`
- Env update: `conda env update -n deadlock -f environment.yml --prune --solver=libmamba` (the classic solver hangs)

Environment gotchas:
- The user's system Python 3.11 has `--user` packages that shadow the env's; `PYTHONNOUSERSITE=1` is set in `environment.yml` and the `deadlock` kernel spec. Keep it.
- Windows PowerShell 5.1 can't open paths over 260 chars (the scratchpad path is longer); pipe code to `python -` instead.

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
