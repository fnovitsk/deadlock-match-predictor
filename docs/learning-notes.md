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

## Milestone 1 — Data exploration

Notebook: `notebooks/01_explore_data.ipynb` (its last section lists every decision).

**What EDA is for.** Before modeling, find out what a row *means*, what's broken, what the label looks like, and whether the inputs carry signal. Every surprise found here would otherwise surface as a mysteriously bad (or suspiciously good) model later.

**Base rate.** Team0 wins 49.9% ± 1.4%, so there's no side advantage. A model that always says 50% scores log loss ln 2 ≈ 0.693; that's the floor any real model must beat.

**Confidence intervals.** With n samples and rate p, the 95% interval is about p ± 1.96·√(p(1−p)/n). With ~1,400 games per hero, a hero's win rate is uncertain by ±3–5 points, so most "strong" and "weak" heroes in the sample are indistinguishable from 50%.

**Label leakage.** The last stat snapshot is taken at the instant the match ends, after the winner has destroyed the base. A model trained on it learns "whoever just won, won." It would score brilliantly offline and be useless live. Likewise, `objectives_mask_team*` describe the end of the match. The rule: a feature at time t may only use information knowable at time t.

**Selection bias.** The team ahead in net worth wins more often as the game goes on, but only until ~35 minutes. After that it drops, because matches that last that long are disproportionately ones where the leader failed to close out. Which matches *survive* to a given time is not random, so game time must be a feature.

**Why logistic regression is a sensible baseline.** Win rate vs. net worth lead is S-shaped at every game time, and that's exactly the curve logistic regression fits: P(win) = σ(w·x + b). The slope changes with time (a 10k lead matters more at 15 minutes than at 35), which hints that interactions or a tree model (Milestone 4) will help.

**Data engineering lessons.** Read only the columns you need (Parquet + DuckDB made a 4 GB remote file queryable in seconds). Check how a dataset is updated: here, new rows (even for old matches) land in delta files, so reading only base files would silently miss data. And keep the sample reproducible: hashing `match_id` with a seed gives the same "random" matches every run.
