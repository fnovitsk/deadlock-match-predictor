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

Models (`src/deadlock_predictor/models/`, settings in `configs/model.yaml`):
- `train_baseline` → `models/baseline*.joblib` (sections `baseline`: rungs 1-4, `baseline_heroes`: rungs 5-6); `train_gbm` → `models/gbm*.joblib` and `models/final.json` (pick among 4 cells, simplest within 0.0005); `refit` → `*_refit.joblib` on train + validation; `evaluate_test` → `models/test_metrics.json` (refuses to rerun without `--force`).
- Saved models are dicts with `model`, `columns`, `hero_ids`: predict with `models.data.predict_saved`.
- Baseline = rung 3 (all diffs × time spline, LR, no intercept): validation log loss 0.5489, **test 0.5472**. Final = LightGBM + heroes, refit: validation 0.5183, **test 0.5232**. Always-50%: 0.6931.
- Models must stay symmetric (P(mirrored) = 1 − P): LR via diffs and no intercept; LightGBM (`models.gbm.SymmetricLGBM`) via mirrored training and logit(P) = ½(g(x) − g(mirror x)).
- Heroes: `heroes.parquet` (match_id, hero_id, sign) → signed one-hot `hero_<id>_diff` via `models.data.add_heroes`, hero list from the fitting splits. Hero 86 was released Oct 16 (validation), hero 87 Oct 20 (test): new heroes get no column. Hero strength drifts (validation edge ≈ 0.84× train's), so hero models are overconfident (ECE ~0.011); retrain on recent data.
- `models.evaluate.scores` / `scores_by_time` / `reliability` are the shared metrics.
- **The test split (Oct 19 → end of data) was scored at the end of Milestone 4.** It is used now: new comparisons need a newer period.
- LightGBM uses `n_jobs: 8`: all 16 threads at full load crashed the machine once.
- DuckDB 2.0 (what CI installs) has no `fetchone()` on relations: use `.fetchall()[0]`.
- `t` is unsigned in the data: convert to float before arithmetic or nearest-value lookups.
- `refit --all` → `*_deploy.joblib` on all splits: what the curves use. Nothing is left to score them on.

Curves (`python -m deadlock_predictor.viz.curve <match_id>` → `reports/curves/`, gitignored):
- `data.match.match_rows`: local dump first (`local_rows`), else `api.deadlock-api.com/v1/matches/{id}/metadata` converted to dump rows (`api_rows`); `register` loads rows into DuckDB with VARCHAR[] list types, then the training feature code runs unchanged (`filter_matches(keep_failing=True)` keeps a match that fails a filter and reports it).
- API vs dump: team 0/1 → 'Team0'/'Team1'; `team_objective_id` 0 Core, 1-4 Tier1Lane1-4, 5-8 Tier2Lane1-4, 9 Titan, 10-11 TitanShieldGenerator1-2, 12-15 BarrackBossLane1-4; match_mode 4 = Ranked, game_mode 1 = Normal. The API has no lobby badge: `lobby_badge` recomputes it from players' `initial_display_rank` (mean on the tier·6 + subtier scale, unranked skipped, rounded half up). Features matched the dump on every sampled match.
- Events use the in-game names (`viz.curve.EVENT_NAMES`): Tier1 = Lane Guardian, Tier2 = Walker, BarrackBoss = Base Guardian, TitanShieldGenerator = Shrine, Titan = Patron Downed. Team0 is most likely the Hidden King and Team1 the Archmother (deadlock-api.com lists them in that order, with the same small side edge as Team0); not confirmed in the API docs.
- `/v1/matches/recently-fetched` lists recent matches (for fresh, never-seen data).
- `models.data.predict_saved(saved, df, heroes=...)` takes hero picks directly for matches not in `heroes.parquet`.

Live games and dashboard (M6):
- `/v1/matches/active`: per team `net_worth` and `objectives_mask`, per player `hero_id`, `abandoned`; plus `start_time`. No kills, levels, damage, mid-boss or rank. Ranked 6v6 = `match_mode_parsed == "Ranked"` and `game_mode_parsed == "KECitadelGameModeNormal"`. Mask bit i set = objective i still standing (verified 63,009/63,009); unused lane-2 bits are always set.
- `data.live.live_state` builds the live model's inputs. Game time = `now − start_time − PREGAME_S` (210 s of pre-game; found by `live.score`'s clock check). The live model (`gbm_live`: t, net worth, 5 objective diffs, heroes): validation 0.5259; deploy version scored live 0.503 on 421 games.
- `live.collect` logs polls to `data/live/<UTC date>.parquet`; `live.score` writes `matches.parquet`, `scored.parquet`, `scores.json` next to them (`load()` reads only the dated logs).
- Dashboard: `streamlit run src/deadlock_predictor/app/dashboard.py` (logic in `app/helpers.py`, tested). Streamlit doesn't reload imported modules: restart it after editing them. Widgets inside the auto-refreshing fragment must not have options that change each poll (they reset).

Match explorer (M7):
- `data.match_detail.fetch_detail` caches API metadata in `data/matches/<id>.json`; parsers: `players`, `player_stats` (~50 stats/snapshot, final included), `deaths` (killer None unless an enemy player), `items`, `buffs`, `paths` (x/y per second: `x_min + x_pos / x_resolution · (x_max − x_min)`, per player).
- The API serves processed matches at once; unprocessed ones go to Steam: 503 "Steam is busy", 429 at 3 requests/hour/IP. Don't retry in a loop.
- Stats `kills` ≠ death list (28 vs 24 in one match): `viz.timeline.features_at` anchors kills to snapshots and adds death-list kills since.
- `features_at` equals the training rows at snapshots (tested). Pass `stats` so building souls (`gold_boss`) step in at the event (`bounty_steps`): without it, early buildings move the curve the wrong way (mediator effect).
- Fights (`viz.fights`): `configs/model.yaml` → `explorer:` (20 s, 3,000 units, 3+ deaths).
- `assets.deadlock-api.com` doesn't resolve on this machine: names fall back to ids unless `data/assets/heroes.json` / `items.json` exist.
- Plotly shapes on a subplot with no traces yet are dropped: pass `exclude_empty_subplots=False`. Static PNGs for checking need `kaleido` (install with `pip --target` into the scratchpad, not the env).

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
