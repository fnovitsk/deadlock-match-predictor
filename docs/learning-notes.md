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

## Milestone 2 — Data pipeline

Code: `python -m deadlock_predictor.data.pipeline`. Check: `notebooks/02_check_pipeline.ipynb`.

**Point-in-time correctness.** A feature at time t may only use what was knowable at t. In practice that meant three rules in `features/snapshots.py`: drop the snapshot taken as the match ends, count an objective or mid-boss kill only if it happened at or before t, and never read end-of-match columns. Each rule has a unit test on a hand-made match where the right answer is obvious. Tests like these are how you keep leakage from creeping back in when the code changes later.

**Why Team0 − Team1 differences.** The question is "who's ahead, and by how much?", so the model gets the gap rather than both teams' totals. That halves the inputs and builds in a symmetry: a 5k lead means the same whichever side has it. One label (`team0_win`) then covers both teams, since P(Team1 wins) = 1 − P(Team0 wins).

**Chronological splits.** The model will always predict games played after the ones it learned from, so the test set should be newer than the training set: train on the oldest 70% of matches, validate on the next 15%, test on the newest 15%. A random split would let the model learn from games played *after* some test games (the meta shifts over a patch), which flatters the score. Splitting by match keeps the ~9 nearly identical snapshots of one game out of two splits. Our own data shows why time matters: Team0 won 50.7% ± 0.2% of training matches but ~50.0% later. The game drifted slightly within one patch, and a chronological split is what exposes drift like that.

**Train, validation, test.** Train is for fitting. Validation is for every decision we make (which features, which model, which settings). Test is looked at once, at the end, to get an honest score. Every look at data while deciding something is a little bit of fitting, which is why notebook 02 analyses only train.

**Redundant features.** Every kill is someone's death, so `kills_diff` and `deaths_diff` are near mirror images. That barely hurts predictions, but a linear model can split one effect between two columns in many ways, so its coefficients become unstable and hard to read. Worth removing for an interpretable baseline.

**Pipelines that can rerun.** Each stage writes to disk, so later stages rerun without redoing earlier ones (feature building takes seconds; downloading takes minutes). The download is split per partition, finished partitions are skipped, and each file is written under a temporary name and renamed when complete, so an interrupted run never leaves a half-written file that looks finished.

**Check your assumptions against the data.** Notebook 01 assumed `average_badge` was each player's rank. Building the team rank gap gave a feature that was exactly 0 everywhere: the column is the lobby average, copied to every player. A feature with no variation is a cheap but reliable sign that something is off.

## Milestone 3 — Baseline model

Code: `python -m deadlock_predictor.models.train_baseline`. Results: `notebooks/03_baseline.ipynb`.

**Logistic regression.** P(win) = σ(w · x), with σ(z) = 1 / (1 + e^−z). Training finds the weights that make the observed outcomes most likely, which is the same as minimizing log loss. It's the right first model when the relationship looks like an S-curve, and its weights can be read.

**Scoring probabilities.** Log loss (average −log of the probability given to what happened) punishes confident mistakes hard; Brier score (squared error) is gentler; calibration asks whether "70%" really means 70%. Accuracy is a weak measure for probabilities: it treats an early 52% and a late 95% the same.

**Build symmetry into the model.** With no intercept and only Team0 − Team1 diffs as inputs (or diffs times something about the whole game), the logit is an odd function: swapping the teams gives exactly 1 − P. We planned to train on mirrored copies of every row, but for this model a mirrored row has exactly the same loss as the original, so mirroring would only count every row twice. A constraint in the model's structure beats one in the data.

**Varying coefficients.** Multiplying each diff by a few smooth spline functions of game time lets its weight change with time while staying a linear model. The curves show a soul is worth ~20× less at 60 minutes than at 5, and that buildings matter more than anything but net worth late.

**A ladder of models.** Adding one idea per rung shows what each is worth: net worth alone gives 92% of the gain (0.693 → 0.560), everything else together brings it to 0.549, and lobby rank adds nothing.

**Prefer the simpler model when the gain is noise.** Rung 4 beat rung 3 by 0.00003 log loss. Keeping extra parameters for a difference that small invites overfitting to the validation set, so the training script picks the simplest rung within 0.0005 of the best.

**Regularization depends on data size.** An L2 penalty guards against fitting noise. With 2.4M rows and ~100 weights, every penalty strength scored the same: there's nothing to fix. It will matter when features multiply (heroes) or data shrinks.

**Correlated features make weights hard to read.** Net worth, level and damage move together, so the model can split one effect among them, and some weights look backwards (`level_diff` is negative mid-game). Trust predictions, not individual weights.

**Watch your dtypes.** Snapshot time is stored as an unsigned integer. pandas' "nearest" lookup subtracts index values, and unsigned subtraction wraps around, so it silently picked the wrong neighbor. Converting to float fixed a plot that looked plausible but was wrong.

## Milestone 4 — LightGBM and heroes

Code: `train_gbm`, `refit`, `evaluate_test` in `deadlock_predictor.models`. Results: `notebooks/04_lightgbm_heroes.ipynb`.

**One-hot encoding.** A hero id is a label, not a quantity, so each hero gets its own column (+1 if Team0 picked it, −1 if Team1 did, 0 otherwise). Signed columns are diffs, so they fit the symmetric setup unchanged.

**Information beats model complexity.** Adding heroes cut validation log loss by 0.030; replacing logistic regression with boosted trees cut it by 0.001. The baseline's time-varying weights already captured the structure, and a better model can't use information it isn't given.

**Gradient boosting.** Many small trees fitted in sequence, each to the errors left by the previous ones. `num_leaves` and `min_child_samples` limit how much each tree can fit; early stopping on validation chooses how many trees. Small trees with big leaves won here: the signal is smooth and the noise large.

**Symmetry for trees.** Trees can't be made symmetric by structure, so we trained on every row plus its mirror and predicted ½(g(x) − g(mirror x)) on the logit scale, which is exactly odd. Unlike for the linear model, mirroring changes what trees learn.

**Trees extrapolate in steps.** LightGBM was a bit better than the linear model early (hero combinations) and a bit worse after 35 minutes, where games are rare: a tree predicts a constant from the few rows in its leaf, while a spline weight continues smoothly.

**SHAP values.** They split one prediction into additive contributions per feature; averaging their size shows what drives predictions at each stage. Heroes move predictions ~0.6 logit at every stage; net worth takes over from 15 minutes.

**Drift and new heroes.** Within one patch, hero win-rate edges in validation were only ~0.84× those in train, and two heroes were released in 8 days (one in validation, one in test). Models fitted on older data overstate hero effects (overconfidence: ECE 0.004 → 0.014), and a new hero has no column at all. A deployed model needs regular retraining.

**Refit before the final test.** Once validation has fixed every setting, its matches are just the newest training data. Refitting on train + validation (same C, same tree count) improved the hero model's test log loss from 0.5274 to 0.5232 and its calibration from 0.018 to 0.011.

**Look at the test set once.** All choices (model, settings, refit, what to report) were made before reading it, and `evaluate_test` refuses to run twice. Final test log loss: baseline 0.5472, LightGBM + heroes 0.5232. The gain (0.024) is smaller than on validation (0.031), and almost all of the difference comes from matches with the hero released during the test period: without it, the gain is 0.030, as on validation.

**Change rules before results, not after.** The rule for choosing between models (simplest within 0.0005) was set before training. LightGBM + heroes passed it by a small margin, and we kept the rule rather than overriding it after seeing the numbers.
