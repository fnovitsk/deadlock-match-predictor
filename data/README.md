# data/

Local data only. Everything here except this file is gitignored.

- `raw/` — files downloaded from the deadlock-api.com dumps, unmodified
- `interim/` — filtered or partially processed tables
- `processed/` — final snapshot/feature tables used for training

Everything here can be regenerated with the pipeline scripts in `src/deadlock_predictor/data/`.
