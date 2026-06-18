# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Hotel-cancellation prediction. The notebook `project.ipynb` is the analyst-facing walkthrough (EDA, cleaning, feature engineering, logistic regression with grid search, evaluation, write-up). The reusable code lives in `src/hotels/` and is imported by the notebook, the (future) FastAPI service, the DVC pipeline, and tests.

Current AUC: test 0.8971, CV 0.8986 with `LogisticRegression(class_weight='balanced', C=1.0, penalty='l1')`.

## Layout

- `project.ipynb` — analyst notebook. Built from `_build_nb.py` (cell list + executor). To regenerate: `.venv/bin/python _build_nb.py`.
- `src/hotels/` — installable package (`pip install -e .` via `pyproject.toml`).
  - `config.py` — column lists, paths, `RANDOM_STATE`, `COUNTRY_MIN_FREQUENCY`, `ADR_MAX`. Single source of truth for hyperparameters and column splits.
  - `data.py` — `load_raw`, `drop_leakage`, `fill_missing`, `drop_junk_rows`, `clean`.
  - `features.py` — `add_engineered_features`, `drop_redundant_columns`, `engineer`.
  - `preprocess.py` — `build_preprocessor` (ColumnTransformer factory).
  - `split.py` — `split_xy`, `make_train_test` (stratified 80/20).
- `docker-compose.yml` — services (currently: `mlflow`).
- `docker/mlflow/Dockerfile` — MLflow tracking server image.
- `scripts/mlflow_smoke.py` — confirms the tracking server is reachable.
- `scripts/run_experiments.py` + `src/hotels/experiments.py` — trains LogReg, RandomForest, HistGBM on the DVC-produced split, logs each to MLflow, registers the winner.
- `dvc.yaml` + `params.yaml` — DVC pipeline (clean → featurize → split → train). `dvc.lock` records the materialised state.
- `src/hotels/stages/` — DVC stage entry points (`clean.py`, `featurize.py`, `split.py`, `train.py`), each runnable as `python -m hotels.stages.<name>`.
- `data/`, `models/`, `reports/` — DVC outputs, gitignored (tracked via `dvc.lock`).
- `hotels.csv` — raw data, tracked by DVC via `hotels.csv.dvc`, not by git.
- `.venv/` — uv-managed virtualenv. Always invoke Python as `.venv/bin/python ...`.

## Common commands

```bash
# Activate env (or invoke .venv/bin/python directly).
source .venv/bin/activate

# Rebuild and execute the notebook.
.venv/bin/python _build_nb.py

# MLflow tracking server (http://localhost:5001).
docker compose up -d mlflow
docker compose down

# MLflow smoke test (logs one run).
.venv/bin/python scripts/mlflow_smoke.py

# DVC pipeline: run all stages that are stale.
.venv/bin/dvc repro

# DVC: see the DAG.
.venv/bin/dvc dag

# DVC: see what changed since the last lock.
.venv/bin/dvc status

# Run the multi-model experiment (LogReg / RF / HistGBM).
# Requires the MLflow server up and the DVC split parquets on disk.
.venv/bin/python scripts/run_experiments.py
```

## MLflow registry

- Tracking server: `http://localhost:5001` (MLflow 3.14.0 in Docker, served with `--serve-artifacts`).
- Experiment: `hotels-cancellation`.
- Registered model name: `hotels-cancellation`.
- Winner is tagged `model_kind` and `auc`, and gets the `production` alias.
- Load from any client with `mlflow.sklearn.load_model("models:/hotels-cancellation@production")`.

## MLflow

The tracking server runs in Docker at `http://localhost:5001` (host port 5001 because macOS occupies 5000 for AirPlay). Backend is SQLite, artifact store is `/mlflow/artifacts` inside the container, both on named docker volumes (`mlflow-store`, `mlflow-artifacts`). The server is started with `--serve-artifacts` so the client never touches the artifact filesystem directly. Clients should set `MLFLOW_TRACKING_URI=http://localhost:5001`.

## Project-specific constraints

These are non-obvious decisions baked into the notebook and modules. Treat them as ground truth.

- **Target leakage, drop first**: `reservation_status` and `reservation_status_date` leak `is_canceled`. They are dropped in `data.drop_leakage`. Leaving them in produces ~100% accuracy and a meaningless model.
- **Split before scaling, always**: fit `StandardScaler`/`OneHotEncoder` inside a `Pipeline` after the stratified 80/20 split (`split.make_train_test`). Scaling before splitting leaks test statistics into training.
- **High-cardinality `country`**: 178 values. `OneHotEncoder(min_frequency=50, handle_unknown="ignore")` (in `preprocess.build_preprocessor`) buckets the long tail and tolerates unseen test categories.
- **Class imbalance ~37/63**: use `class_weight="balanced"`. Headline metrics are ROC AUC + recall on the cancelled class, not accuracy. The "never cancels" baseline already hits ~63%.
- **`deposit_type=Non Refund` is a known data artefact**: it shows a 99.4% cancel rate that's backwards from intuition (almost certainly retroactive logging). Flag as a caveat in any write-up; do not build a narrative on it.
- **Missing-value handling is column-specific** (`data.fill_missing`): `children` filled with 0 (only 4 NaNs), `country` filled with "Unknown", `agent`/`company` collapsed to `has_agent`/`has_company` binary flags with the original IDs dropped.
- **Junk-row filters** (`data.drop_junk_rows`): drop rows where `adults + children + babies == 0`; drop the `adr` outlier (~5400) by filtering `0 <= adr < 1000`.
- **Engineered features** (`features.add_engineered_features`): `total_nights` (weekend + week), `total_guests`, `is_family`, `room_changed`. The arrival-date split columns (year/week_number/day_of_month) and the two room-type columns are dropped after engineering; `arrival_date_month` is kept as a categorical for seasonality.
- **`room_changed` is leakage-adjacent**: it can only be 1 if the guest actually showed up. Keep it in the model but flag in any interpretation as not a real-world causal driver.
- **Interpretation expectations**: positive coefs on `lead_time`, `previous_cancellations`, `deposit_type=Non Refund`, `country_PRT`. Negative on `required_car_parking_spaces` (strongest single signal), `total_of_special_requests`, `is_repeated_guest`, `distribution_channel_GDS`.

## Data source

tidytuesday `hotels.csv`. Currently committed via DVC (TBD in Phase 3) rather than git.
