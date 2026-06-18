# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repo currently contains only `README.md` — planning notes for a hotel-cancellation logistic-regression project. No notebook, source files, dependency manifest, or data file exists yet. The deliverable described in the notes is a single Jupyter notebook (charts inline) trained on the tidytuesday `hotels.csv` (~119k rows, two Portuguese hotels, 2015–2017).

## Project-specific constraints

These are non-obvious decisions already made in `README.md` — treat them as ground truth rather than re-debating them:

- **Target leakage — drop first**: `reservation_status` and `reservation_status_date` leak the target (`is_canceled`). They must be dropped before any EDA/modeling step. Leaving them in produces ~100% accuracy and a meaningless model.
- **Split before scaling, always**: fit `StandardScaler`/`OneHotEncoder` inside a `Pipeline` after the stratified 80/20 split. Scaling before splitting leaks test statistics into training.
- **High-cardinality `country`**: 170+ values. Use `OneHotEncoder(min_frequency=..., handle_unknown="ignore")` to bucket the long tail and tolerate unseen test categories.
- **Class imbalance ~37/63**: use `class_weight="balanced"`. Headline metric is **ROC AUC + recall on the cancelled class**, not accuracy — the "never cancels" baseline already hits ~63%.
- **`deposit_type=Non-Refund` is a known data artefact**: it shows an absurdly high cancel rate that's backwards from intuition. Flag it as a caveat in the write-up; do not build a narrative on it.
- **Missing-value handling is column-specific**: `children` → `fillna(0)` (only 4 NaNs); `country` → `"Unknown"`; `agent`/`company` → convert to `has_agent`/`has_company` binary flags, drop the originals (do not impute the IDs).
- **Junk-row filters**: drop rows where `adults + children + babies == 0`; drop the `adr` outlier (~5400) by filtering `adr < 1000`.
- **Engineered features**: `total_nights` (weekend + week nights), `total_guests`, `is_family` (kids or babies > 0), `room_changed` (`reserved_room_type != assigned_room_type`). Drop the two room-type columns after deriving `room_changed`, plus the split-out arrival date pieces (year / week_number / day_of_month). Keep `arrival_date_month` as a categorical for seasonality.
- **Model + tuning**: `LogisticRegression` with `GridSearchCV` over `C` (0.01 … 10) and `l1`/`l2` penalty (`liblinear` or `saga` solver), scored on `roc_auc`, 5-fold stratified.
- **Interpretation**: pull `coef_`, `exp()` into odds ratios, plot top 15 by absolute value. Expected signs — positive: `lead_time`, `previous_cancellations`, `deposit_type`, `market_segment`. Negative: `total_of_special_requests`, `required_car_parking_spaces`, `is_repeated_guest`.

## Data source

tidytuesday `hotels.csv`. The notes suggest loading from the raw GitHub URL on first run and caching locally for faster reruns.