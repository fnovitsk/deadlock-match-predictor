# Project map

The starting point for a new session: what the project is, where it stands, what every file and folder is for, how to rebuild it, and what's open. `CLAUDE.md` has the detailed data facts and gotchas, and `docs/learning-notes.md` has the reasons behind each decision.

## What it is
A win-probability model for Deadlock. Given the state of a ranked 6v6 match at time t (souls, kills, buildings, mid-boss, heroes, rank, game time), it predicts P(Hidden King wins).

The Hidden King is Team0 and the Archmother is Team1; both are drawn in their own colours. The app is one Streamlit page: enter a finished match id to see:
- the scoreboard;
- the win-probability curve at every snapshot, building and fight, with objective icons;
- what moved the curve, and the key moments;
- each player's souls, level, damage and healing.

## Where it stands (Oct 30, 2026)
| # | Milestone | Result |
|---|---|---|
| 0–2 | Setup, exploration, data pipeline | Ranked 6v6 since the Sep 29 patch → `snapshots.parquet`, split 70/15/15 by date |
| 3 | Baseline | Symmetric logistic regression: test log loss 0.5472 |
| 4 | LightGBM + heroes | Symmetric LightGBM: **test 0.5232** (always-50% is 0.6931). Test split used once |
| 5 | Curves | `viz.curve` CLI: PNG for any match (dump or API) |
| 6 | Dashboard + live model | Live model kept (validation 0.526; 0.503 on 421 real live games). The live page was later removed |
| 7 | Match explorer | Curve points at every event (`features_at`), fights, building-souls fix |
| 8 | Match page UI | Dark theme, scoreboard, icon chart, key moments, players. **PR #9** (`feat/match-ui`) |

## How the app works
1. `app/dashboard.py`: a match id from the text box goes to `viz.explorer.load_match`.
2. `load_match`:
   - fetches the API metadata (`data.match_detail.fetch_detail`, cached in `data/matches/`);
   - turns it into the training rows (`data.match.api_rows` → `viz.curve.match_features`);
   - scores the deploy models at every snapshot and event (`viz.timeline.event_points`);
   - groups deaths into fights (`viz.fights`);
   - adds hero names, colours and portraits (`data.assets`).
3. `app/match_page.py` draws the `Match` with `viz.charts` (theme `viz.theme`, icons `viz.icons`, key moments `viz.moments`).

## Files and folders

### Tracked
| Path | What |
|---|---|
| `README.md` | Overview, setup, commands |
| `CLAUDE.md` | Facts and gotchas for Claude sessions |
| `docs/learning-notes.md` | ML concepts and decisions, milestone by milestone |
| `docs/project-map.md` | This file |
| `configs/data.yaml` | Data scope (ranked 6v6 since the patch), sample and pipeline settings |
| `configs/model.yaml` | Baseline ladders, LightGBM grids (`gbm`, `gbm_heroes`, `gbm_live`), final pick, refit, fight settings (`explorer:`) |
| `environment.yml`, `pyproject.toml` | Conda env `deadlock`; package, pytest and ruff settings |
| `.pre-commit-config.yaml`, `.github/workflows/ci.yml` | ruff + file checks; CI runs ruff + pytest on PRs |
| `.streamlit/config.toml` | Dark theme |
| `tests/` | pytest, one file per module, all offline |

### `src/deadlock_predictor/`
| Module | Role |
|---|---|
| `data/dumps.py` | Manifest, `USER_AGENT`, DuckDB connection to the dumps |
| `data/columns.py` | Dump column lists |
| `data/extract.py`, `data/pipeline.py`, `data/split.py` | Pipeline: raw partitions → snapshots + heroes → date split (CLI: `data.pipeline`) |
| `data/sample.py` | M1 exploration sample (CLI; wrote `data/interim/sample_ranked.parquet`) |
| `data/match.py` | One match's rows from the dump or the API (`match_rows`, `api_rows`, `lobby_badge`) |
| `data/match_detail.py` | API metadata cache + parsers (`players`, `player_stats`, `deaths`) |
| `data/assets.py` | Hero names, colours, portraits and rank badges from `/v1/assets` |
| `data/live.py` | Live model inputs from `/v1/matches/active` (kept, not in the app) |
| `features/snapshots.py`, `features/heroes.py` | Training features and label; hero picks |
| `models/baseline.py`, `train_baseline.py` | Symmetric LR ladder (CLI) |
| `models/gbm.py`, `train_gbm.py` | `SymmetricLGBM`, grids, final pick (CLI) |
| `models/data.py`, `evaluate.py` | Loading splits, heroes, `predict_saved`; shared metrics |
| `models/refit.py`, `evaluate_test.py` | Refit / deploy models (CLI); one-time test scoring |
| `viz/curve.py` | Model loading, `match_features`, event names, static PNG curve (CLI) |
| `viz/timeline.py`, `viz/fights.py` | Curve points between snapshots; fight clustering |
| `viz/explorer.py` | `Match` + `load_match`, `state_at`, `player_table` |
| `viz/charts.py`, `theme.py`, `icons.py`, `moments.py` | The match page's charts, theme, objective icons, key moments |
| `app/dashboard.py`, `app/match_page.py`, `app/helpers.py` | The Streamlit app |
| `live/collect.py`, `live/score.py` | Log live games; score the live model once they end (CLI, kept) |

### Local only (gitignored)
| Path | What |
|---|---|
| `data/raw/` (~850 MB) | Downloaded dump partitions; `data.pipeline --force` rebuilds them |
| `data/processed/` | `snapshots.parquet`, `heroes.parquet`, `summary.json` |
| `data/matches/` | API metadata cache; safe to delete |
| `data/assets/` | Asset JSON + cached images; delete to refresh after a game update |
| `data/interim/` | The user's executed analysis notebooks (M1–M7) and the M1 sample |
| `data/live/` | The Oct 26 live log and its scores |
| `models/` | Every trained model; the app uses `baseline_deploy`, `gbm_heroes_deploy`, `final.json` |
| `reports/curves/` | `viz.curve` output |
| `.claude/launch.json` | Claude's preview server config (`dashboard`, port 8501) |

## Rebuilding everything
```bash
python -m deadlock_predictor.data.pipeline            # data/raw → data/processed
python -m deadlock_predictor.models.train_baseline    # LR ladders
python -m deadlock_predictor.models.train_gbm         # LightGBM grids + final pick (add --section gbm_live for the live model)
python -m deadlock_predictor.models.refit --all       # refit + deploy models
streamlit run src/deadlock_predictor/app/dashboard.py
```

## Next stage: open threads
- **Fresh data and drift.**
  - The test period is used up (scored once in M4), so new comparisons need data newer than Oct 2026.
  - Hero strength drifts: the hero models are a little overconfident (ECE ~0.011). Retraining on recent data, or weighting recent matches, is the obvious next model step.
- **New heroes** (86, 87) get no column until a retrain includes them.
- **Live predictions:**
  - The model works (0.503 live), but `/v1/matches/active` has no history and is 2–4 min stale.
  - The live broadcast (`/v1/matches/demo/live/query`) has the full model's inputs at ~22 s delay, but only from when you start watching.
  - A live feature would need an always-on collector or the broadcast.
- **The match page** could add the removed views back from git history (map replay, builds, fights tab) if wanted.
- **PR #9** (`feat/match-ui`) is open; delete the branch after it merges.
