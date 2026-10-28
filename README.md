# Deadlock Match Predictor

Estimates each team's probability of winning a [Deadlock](https://store.steampowered.com/app/1422450/Deadlock/) match at any point in time, in the spirit of Valve's Dota 2 win-probability graph.

The model is trained on historical match data: every in-game stat snapshot (net worth, kills, objectives, heroes, game time) becomes a training example labeled with whether that team went on to win. Running the model over every snapshot of one match produces a win-probability curve.

> **Status:** Milestone 6 (dashboard and live predictions). See the roadmap below.

## Roadmap

| # | Milestone | Output |
|---|-----------|--------|
| 0 | Project setup | Repo, environment, conventions |
| 1 | Data exploration | `notebooks/01_explore_data.ipynb` |
| 2 | Data pipeline | Clean snapshot table (Parquet) |
| 3 | Baseline model | Logistic regression + calibration |
| 4 | Better model | Richer features + LightGBM |
| 5 | Win-probability curves | Plot for any match ID |
| 6 | Dashboard and live predictions | Streamlit app, live model, collector and scorer |
| 7 | Match explorer | Interactive per-player view of a match |

## Setup

Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and git.

```bash
git clone https://github.com/fnovitsk/deadlock-match-predictor.git
cd deadlock-match-predictor
conda env create -f environment.yml --solver=libmamba
conda activate deadlock
python -m ipykernel install --user --name deadlock --display-name "Python (deadlock)" --env PYTHONNOUSERSITE 1
pre-commit install
pytest
```

To update the environment after `environment.yml` changes:

```bash
conda env update -f environment.yml --prune --solver=libmamba
```

`--solver=libmamba` is much faster than conda's classic solver (the default before conda 23.10). `PYTHONNOUSERSITE=1` (set by `environment.yml` and in the kernel spec) keeps packages pip-installed with `--user` for a system Python from shadowing the environment's own.

## Building the dataset

```bash
python -m deadlock_predictor.data.pipeline
```

Downloads every ranked 6v6 match since the last major patch (one file per partition in `data/raw/`; reruns skip finished partitions) and writes `data/processed/snapshots.parquet`: one row per match and snapshot time, with Team0-minus-Team1 features, the label, and a chronological train/validation/test split. It also writes `data/processed/heroes.parquet` (each team's hero picks). `notebooks/02_check_pipeline.ipynb` checks the result.

## Training the baseline

```bash
python -m deadlock_predictor.models.train_baseline
```

Fits a ladder of symmetric logistic regressions on the training split, tunes them on validation, and saves the best to `models/`. Validation log loss: 0.549 (always-50% scores 0.693). `notebooks/03_baseline.ipynb` explains the results.

## Training the final model

```bash
python -m deadlock_predictor.models.train_baseline   # both logistic ladders (with and without heroes)
python -m deadlock_predictor.models.train_gbm        # LightGBM grids, then picks the final model
python -m deadlock_predictor.models.refit            # refit baseline and final model on train + validation
python -m deadlock_predictor.models.evaluate_test    # score both on the test split, once
```

The final model is LightGBM with hero picks, made exactly symmetric by training on team-swapped copies and averaging both views. Test log loss: **0.523** (baseline 0.547). LightGBM training resumes where it stopped if interrupted. `notebooks/04_lightgbm_heroes.ipynb` explains the results.

## Drawing a curve

```bash
python -m deadlock_predictor.models.refit --all      # once: refit the final model and baseline on every match
python -m deadlock_predictor.viz.curve <match_id>    # saves reports/curves/<match_id>.png
```

Works for any ranked 6v6 match: from the local dump if it's there, otherwise fetched from the deadlock-api.com API (including matches played today). The chart shows P(Hidden King wins) at each stat snapshot (every 3–5 minutes) for the final model and the baseline, the buildings (Lane Guardians, Walkers, Base Guardians, Shrines, Patron) and mid-boss claims in between, and what moved the prediction since the previous snapshot. `notebooks/05_win_curves.ipynb` shows examples and checks the curves.

## Dashboard and live games

```bash
python -m deadlock_predictor.models.train_gbm --section gbm_live   # once: the live model
python -m deadlock_predictor.models.refit --all                    # once: deploy versions
streamlit run src/deadlock_predictor/app/dashboard.py
```

Two pages: **Match lookup** draws the curve of any finished match, and **Live games** lists ranked games in progress with the live model's win probability, refreshed every minute. Click a game or type a match id to follow its curve. The live model sees only what a game in progress shows (souls, buildings, heroes): validation log loss 0.526 against 0.518 for the full model.

To check live predictions against real results:

```bash
python -m deadlock_predictor.live.collect --minutes 120   # log ranked games in progress once a minute
python -m deadlock_predictor.live.score                   # after they finish: score every logged prediction
```

On 421 games: live log loss **0.503**, well calibrated. `notebooks/06_live.ipynb` explains the results, including a 3.5-minute clock offset in the live data.

## Project layout

```
configs/                 settings (data scope, features)
data/                    local data, not committed (raw/, interim/, processed/)
docs/learning-notes.md   ML concepts and decisions, milestone by milestone
models/                  trained models, not committed
notebooks/               exploration; imports reusable code from src/
src/deadlock_predictor/  package: data/, features/, models/, viz/, live/, app/
tests/                   pytest
```

## Data

Match data comes from the public [deadlock-api.com](https://deadlock-api.com/) [data dumps](https://deadlock-api.com/data-dumps). Thanks to the deadlock-api project for making it available. This project is not affiliated with Valve.

## License

[MIT](LICENSE)
