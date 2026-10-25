# Deadlock Match Predictor

Estimates each team's probability of winning a [Deadlock](https://store.steampowered.com/app/1422450/Deadlock/) match at any point in time, in the spirit of Valve's Dota 2 win-probability graph.

The model is trained on historical match data: every in-game stat snapshot (net worth, kills, objectives, heroes, game time) becomes a training example labeled with whether that team went on to win. Running the model over every snapshot of one match produces a win-probability curve.

> **Status:** Milestone 3 (baseline model). See the roadmap below.

## Roadmap

| # | Milestone | Output |
|---|-----------|--------|
| 0 | Project setup | Repo, environment, conventions |
| 1 | Data exploration | `notebooks/01_explore_data.ipynb` |
| 2 | Data pipeline | Clean snapshot table (Parquet) |
| 3 | Baseline model | Logistic regression + calibration |
| 4 | Better model | Richer features + LightGBM |
| 5 | Win-probability curves | Plot for any match ID |
| 6 | (Later) | Dashboard, live predictions |

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

Downloads every ranked 6v6 match since the last major patch (one file per partition in `data/raw/`; reruns skip finished partitions) and writes `data/processed/snapshots.parquet`: one row per match and snapshot time, with Team0-minus-Team1 features, the label, and a chronological train/validation/test split. `notebooks/02_check_pipeline.ipynb` checks the result.

## Training the baseline

```bash
python -m deadlock_predictor.models.train_baseline
```

Fits a ladder of symmetric logistic regressions on the training split, tunes them on validation, and saves the best to `models/`. Validation log loss: 0.549 (always-50% scores 0.693). `notebooks/03_baseline.ipynb` explains the results.

## Project layout

```
configs/                 settings (data scope, features)
data/                    local data, not committed (raw/, interim/, processed/)
docs/learning-notes.md   ML concepts and decisions, milestone by milestone
models/                  trained models, not committed
notebooks/               exploration; imports reusable code from src/
src/deadlock_predictor/  package: data/, features/, models/, viz/
tests/                   pytest
```

## Data

Match data comes from the public [deadlock-api.com](https://deadlock-api.com/) [data dumps](https://deadlock-api.com/data-dumps). Thanks to the deadlock-api project for making it available. This project is not affiliated with Valve.

## License

[MIT](LICENSE)
