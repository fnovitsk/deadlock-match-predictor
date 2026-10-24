# Learning notes

Concepts and decisions, milestone by milestone.

## Milestone 0 — Project setup

**The prediction problem.** Each match contributes many training examples: one per stat snapshot. An example is the game state at time t (net worth gap, kills, objectives, heroes, game time), labeled 1 if team 0 eventually won. The model learns P(team 0 wins | state at t), and team 1's probability is 1 minus that. Plotting it across a match's snapshots gives the win-probability curve.

**Why `src/` plus notebooks.** Notebooks are good for exploring but hard to test, reuse, or review in git. Code that will be used more than once goes into the `deadlock_predictor` package; notebooks import it. Installing the package in editable mode (`pip install -e .`) means edits in `src/` take effect without reinstalling.

**Reproducible environment.** `environment.yml` describes every dependency, so anyone (including future you) can rebuild the same environment with one command.

**Git hygiene.**
- Data and models are large and can be regenerated, so they're gitignored. The repo holds the code that produces them.
- `nbstripout` removes notebook outputs on commit so diffs show code changes, not megabytes of plots.
- `pre-commit` runs checks (ruff lint/format, large-file guard) before each commit is created.
- `main` always works. Each milestone is developed on its own branch and merged through a pull request.

**Preview: leakage.** Snapshots from the same match are highly correlated. If one match's minute 10 is in training and its minute 20 is in test, the model effectively sees the answer. So we will split by match, and by date (train on older matches, test on newer ones) to mimic predicting the future.
