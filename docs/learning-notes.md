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

## Milestone 5 — Win-probability curves

Code: `python -m deadlock_predictor.viz.curve <match_id>`. Results: `notebooks/05_win_curves.ipynb`.

**Training–serving skew.** A model is only as good as the features it gets at prediction time. Matches not in our dump come from the API, which encodes the same data differently (numeric codes, no lobby badge). We converted it to the dump's format so the *same* feature code runs, and checked that both paths give identical features on random matches. A subtle example: Python's `round` rounds 62.5 to 62 (to even), while the data's source rounds half up; one in 20 sampled matches differed until that was fixed.

**Deploy models.** Once every choice is made and the test score recorded, the final model is refit on *all* the data, including test. It can't be scored any more, but it knows the newest heroes and strengths. On matches played after the dump ended, it beat the train + validation version in all three samples we tried, by 0.001–0.014 log loss.

**Explaining a single prediction.** SHAP contributions add up to the logit, so the change between two snapshots splits exactly into what moved it: net worth, fights, objectives, heroes. That's what the curve's lower panel shows.

**Calibration you can see: comebacks.** If a calibrated curve shows 10%, that team should win about 10% of the time from there. Because a curve usually jumps *past* a threshold rather than landing on it, the fair expectation is the probability at the first snapshot below it. On the test split, teams written off at ≤10% won 5.2% of the time, against 5.4% expected. (A calibrated probability is a fair bet whenever you look at it; in probability theory such a process is called a martingale.)

**Small samples swing.** Three samples of ~150 brand-new matches gave baseline log losses of 0.549–0.587. Compare models on the *same* matches; differences between models are much more stable than either score alone, and ECE on a few thousand rows is mostly noise.

## Milestone 6 — Dashboard and live predictions

Code: `streamlit run src/deadlock_predictor/app/dashboard.py`, `python -m deadlock_predictor.live.collect`, `python -m deadlock_predictor.live.score`. Results: `notebooks/06_live.ipynb`.

**Train on what you'll have at prediction time.** A game in progress shows only souls, buildings and heroes. Feeding the full model zeros or guesses for kills, levels and damage would put it off its training distribution. So the live model is trained on just the live inputs. It costs little: validation log loss 0.5259 vs 0.5183, because souls already carry most of what kills and farm say.

**Decoding an undocumented field.** The live API's `objectives_mask` is a bitmask. The hypothesis "bit *i* set = objective *i* standing" was checked against the dump's per-objective destroyed times: 63,009 of 63,009 agreed. Check an encoding against an independent source before building on it.

**Score predictions on the future, for real.** A collector logged 430 games once a minute; after they ended, a scorer joined each logged prediction with the result. That's the most honest test there is: no possible leakage, a different data source, and the exact model the dashboard serves. Live log loss: 0.503, well calibrated.

**Check inputs, not just outputs.** The live game time was computed as `now − start_time`. Comparing live souls with the post-game souls of the same match at shifted times showed the clock was 3.5 minutes fast: `start_time` includes the pre-game. The model saw every game as older, and so more decided, than it was. Subtracting 210 s cut live log loss from 0.543 to 0.503, more than any model change. The offset was measured on the same games it was scored on, but it's one constant with a physical meaning, not a fit to outcomes.

**Why the live score beat validation.** The scored model is the deploy version, trained on data up to Oct 25, so its hero strengths are current. Validation scores the train-only model, whose hero strengths are weeks old. With 421 games from one evening, the score is good to about ±0.01.

**Streamlit state.** A Streamlit script reruns top to bottom on every interaction, and widgets keep their value only while their identity (label, options) stays the same. A dropdown whose options change every refresh resets itself. The fix is to keep the choice in `st.session_state` and in widgets that don't change (a text box, a row selection read in a callback).

## Milestone 7 — Match explorer

Code: `viz/explorer.py` (dashboard **Match lookup**), `viz/timeline.py`, `viz/fights.py`, `data/match_detail.py`. Results: `notebooks/07_match_explorer.ipynb`.

**Predicting between training points.** The model was trained only at stat snapshots, but the explorer wants a probability at every building and fight. Some inputs are exact at any second (buildings, mid-boss, kills); others exist only at snapshots and are interpolated (souls, levels, damage). The check that keeps this honest: at snapshot times the rows equal the training rows exactly (difference 0.0 on 29 matches). In between, interpolation uses the *next* snapshot, so those points are a best estimate of the state, not what a live viewer could have known. Fine for a post-game view, wrong for a live one.

**Mediators: a feature's effect is "given the others".** At first, early buildings *lowered* the scoring team's chances 94% of the time. The model learned a building's effect with its soul bounty already in `net_worth_diff`, so "one more Guardian, same souls" reads as "worse everywhere else". Souls are a mediator: on the path from the building to winning. Feeding a model a cause without its mediator gives a misleading answer. Adding each building's souls (`gold_boss`) at the second it falls fixed the direction: 63% → 93% of events now move the curve toward the team that scored.

**When two fields disagree, anchor to what the model saw.** The API's per-snapshot `kills` and its death list disagree (28 vs 24 kills for one team in one match). The curve uses the snapshot value (what the model trained on) plus exact kills since that snapshot, so it changes at the right second and stays exact at snapshots.

**Clustering with simple rules.** Teamfights are deaths within 20 s of each other and within 3,000 units of the fight's centre, with several fights open at once so different lanes stay apart. The thresholds come from the data: half of all deaths come within 14 s of the previous one. A transparent rule you can explain beats a clustering algorithm nobody can tune.

**Fights act through souls.** Fights move the curve toward their winner, but only slightly at the time (~0.02 for 4+ deaths): their value reaches the model through the souls they give, which interpolation spreads out. Stepping kill souls (`gold_player`) in at each kill, like building souls, is the next improvement.

**A 2% team winning isn't a model failure by itself.** One match went from 2% (at exact snapshot points) to a win. A calibrated 2% should happen 1 time in 50; calibration is checked on many matches (notebook 05), not argued from one.

## Milestone 8 — Match page UI

Code: `app/match_page.py`, `viz/charts.py`, `viz/moments.py`, `viz/theme.py`.

**One question per chart.** The M7 figure stacked four rows into one image. The page now gives each question its own chart, at a readable size:
- who's winning, and when it changed (the curve);
- why (what moved it, key moments);
- who did what (the player timeline);
- where (the map).

One shared time range keeps them in sync, which is what the single figure was for.

**Encodings.**
- **Side → colour:** amber for the Hidden King, sapphire for the Archmother.
- **Player → the hero's own colour:** 12 colours you can still tell apart.
- **Event type → marker shape:** ● kill, ✕ death, item icon, ◆ buff.
- **Size → importance:** big icons only for items of 3,200+ souls; ticks for the rest.
- **Lines connect cause and effect:** the kill web draws each kill from killer to victim.

**Check the map before you trust it.** The minimap image lined up with player positions exactly, so every street was walked. The asset's own building positions didn't line up at all. So buildings are placed where the data says they fall: the centre of the attackers' group in the 10 s before each building dies, then the densest group of those estimates across 32 matches. Taking the densest group rather than the mean ignores matches where the attackers' main group was elsewhere. That's the same "agreement over averaging" idea as fight clustering.

**Summaries that explain themselves.** Key moments merge curve steps within 60 s, rank them by how much they moved P, and name the cause from the model's own contributions ("mostly net worth, objectives"). It's the model's explanation, not ground truth. But it's the same SHAP decomposition that adds up exactly to the curve.

**Performance is part of the design.** The replay had 561 frames. Building them as validated Plotly objects took 12 s. Plain dicts plus positions held as a players × seconds array took 3 s. Images are embedded as data URIs, so the page doesn't depend on the asset host once cached.

**Less can be more.** After seeing the full page, the player timeline, final builds, the fights tab and the map replay were removed. The page kept the scoreboard, the curve with its key moments, and per-player stats. Building a view is the cheapest way to find out whether it earns its place. Everything removed is in the git history of `feat/match-ui` if it's ever wanted back.

**Why the live page was dropped.** A game watched from minute 20 showed a flat curve for minutes 20–21.5 only, while the in-game clock said 24. Checking it end to end:
- **The model was fine:** live log loss 0.503 on 421 finished games.
- **You can't backfill a stream you didn't record.** `/v1/matches/active` gives only the current state. The live broadcast (what spectators watch) keeps its first fragment and then only the last ~90 s. deadlock-api's live SQL query streams from "now". So a game's curve starts when something starts recording it. The only way round that is a collector running all the time.
- **Observation time vs wall time.** The API sits behind a CDN cache (`max-age=120, stale-while-revalidate=120`), so each version is 15–120 s old when read. We stamped rows with our own clock, so:
  - minute polls of a 2-minute cache logged each version twice, at two different times (49% of rows were repeats), which drew flat steps;
  - the cache age (~65 s on average) ended up inside the fitted `PREGAME_S = 210`.

  A feature row should carry the time its values were *measured*: here, the response's `Last-Modified` header.
- **A product decision follows from what the data source allows.** A live view that starts mid-game, minutes behind the real clock, wasn't worth its cost in this app. The live model and its tooling (`data.live`, `live.collect`, `live.score`, notebook 06) stay, in case a better live source appears.
