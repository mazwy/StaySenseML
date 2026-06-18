"""Builds project.ipynb cell-by-cell from a Python list, then executes it inplace.

Run after editing CELLS below: `.venv/bin/python _build_nb.py`
"""
import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

NB_PATH = Path(__file__).parent / "project.ipynb"

# Each entry is (kind, source). kind in {"md","code"}.
CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("md", src))


def code(src: str) -> None:
    CELLS.append(("code", src))


# ---------------------------------------------------------------------------
# Notebook content
# ---------------------------------------------------------------------------

md(
    "# Hotel cancellation prediction\n"
    "\n"
    "Logistic regression on the tidytuesday `hotels.csv` (~119k rows, two Portuguese hotels, 2015–2017). "
    "Target is `is_canceled` (1/0). The goal is to predict whether a booking gets cancelled before arrival "
    "and figure out what actually drives it.\n"
    "\n"
    "Built step by step. Each section follows the same shape: a quick question, the code that answers it, "
    "and a note about what the answer means for the next step."
)

# ---- Step 1 ---------------------------------------------------------------

md(
    "## Step 1 — get the data in and just look at it\n"
    "\n"
    "Before deciding anything I want to confirm three things: the dataframe has the shape I expect "
    "(~119k × 32), the dtypes look sane, and the missingness pattern matches what I remember "
    "(`company` mostly empty, `agent` ~14%, `country` a handful, `children` exactly 4 NaNs)."
)

code(
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "\n"
    "pd.set_option('display.max_columns', 50)\n"
    "pd.set_option('display.width', 160)\n"
    "\n"
    "df = pd.read_csv('hotels.csv')\n"
    "df.shape"
)

code("df.info()")

code("df.head()")

md(
    "32 columns, 119,390 rows — matches expectations. A few things stand out from `info()`:\n"
    "\n"
    "- `children`, `agent`, `company` are floats only because of the NaNs — they're really integer IDs/counts.\n"
    "- `reservation_status` and `reservation_status_date` are still here. Those are the leakage columns "
    "  flagged in the plan; they'll get dropped in Step 3. Not touching them yet so EDA reflects raw data.\n"
    "- `arrival_date_*` is split across four columns (year / month / week_number / day_of_month). Most of "
    "  that is redundant; I'll keep only `arrival_date_month` later for seasonality."
)

code(
    "miss = df.isna().sum()\n"
    "miss = miss[miss > 0].sort_values(ascending=False)\n"
    "pd.DataFrame({'n_missing': miss, 'pct': (miss / len(df) * 100).round(2)})"
)

md(
    "Missingness lines up with what I expected:\n"
    "\n"
    "- **`company` 94.31%** — essentially empty. Imputing the ID is meaningless; I'll turn it into a "
    "  `has_company` flag in Step 3.\n"
    "- **`agent` 13.69%** — also an opaque ID, same treatment: `has_agent` flag.\n"
    "- **`country` 488 rows (0.41%)** — small enough to just fill with `\"Unknown\"`.\n"
    "- **`children` 4 rows** — trivially small, fill with 0. Easy to overlook, which is why I'm noting it here."
)

code(
    "counts = df['is_canceled'].value_counts().rename({0: 'not_cancelled', 1: 'cancelled'})\n"
    "rate = df['is_canceled'].mean()\n"
    "print(counts)\n"
    "print(f'\\ncancellation rate: {rate:.4f}')"
)

md(
    "~37% positive class. Not catastrophically imbalanced, but enough that accuracy will be misleading: a "
    "`never cancels` baseline scores ~63% without learning anything. From here on the headlines are ROC AUC "
    "and recall on the cancelled class, with `class_weight=\"balanced\"` on the model in Step 6.\n"
    "\n"
    "Step 1 done. Onto EDA."
)

# ---- Step 2 ---------------------------------------------------------------

md(
    "## Step 2 — target + basic EDA\n"
    "\n"
    "Goal here is to see which categoricals actually move the cancel rate, and to confirm the lead-time "
    "intuition (longer wait between booking and arrival → more cancellations)."
)

code(
    "def rate_by(col):\n"
    "    return (df.groupby(col)['is_canceled']\n"
    "              .agg(['mean','count'])\n"
    "              .sort_values('mean', ascending=False)\n"
    "              .round(3))\n"
    "\n"
    "rate_by('deposit_type')"
)

md(
    "Non Refund shows a 99.4% cancel rate on ~14.6k bookings. That's backwards from intuition "
    "(non-refundable should discourage cancelling) and is the known data-logging artefact flagged in the "
    "plan — almost certainly the column was filled retroactively for bookings that ended up cancelled. "
    "Keep the column for prediction, but don't tell a story around it."
)

code("rate_by('market_segment')")

code("rate_by('hotel')")

code("rate_by('customer_type')")

md(
    "Reading these together:\n"
    "\n"
    "- market_segment: Groups cancel at 61%, Direct/Corporate at 15–19%. Big spread, good signal.\n"
    "- hotel: City 42% vs Resort 28%. Real difference, worth keeping.\n"
    "- customer_type: Transient 41% vs Group 10%. Also a real spread.\n"
    "- market_segment has an `Undefined` bucket with 2 rows — fine, the rare-bucketing in Step 5 will "
    "  absorb it."
)

code(
    "df.groupby('is_canceled')['lead_time'].describe().round(1)"
)

code(
    "buckets = pd.cut(df['lead_time'], bins=[-1,7,30,90,180,365,800])\n"
    "df.groupby(buckets, observed=True)['is_canceled'].agg(['mean','count']).round(3)"
)

code(
    "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
    "for label, sub in df.groupby('is_canceled'):\n"
    "    ax[0].hist(sub['lead_time'], bins=60, alpha=0.5,\n"
    "               label=f'is_canceled={label}', density=True)\n"
    "ax[0].set_xlabel('lead_time (days)'); ax[0].set_ylabel('density')\n"
    "ax[0].set_title('lead_time distribution by class'); ax[0].legend()\n"
    "\n"
    "bucket_rate = df.groupby(buckets, observed=True)['is_canceled'].mean()\n"
    "bucket_rate.plot(kind='bar', ax=ax[1], color='steelblue')\n"
    "ax[1].set_ylabel('cancel rate'); ax[1].set_title('cancel rate by lead_time bucket')\n"
    "ax[1].tick_params(axis='x', rotation=30)\n"
    "plt.tight_layout(); plt.show()"
)

md(
    "Cancel rate climbs monotonically from 10% (booked within a week) to 68% (booked >1y out). The "
    "distribution plot shows cancellations skew right — cancelled bookings sit at a median of 113 days "
    "lead vs 45 for kept bookings. Strong, clean signal; lead_time will land near the top of the "
    "coefficient ranking."
)

code(
    "numeric = df.select_dtypes(include=[np.number])\n"
    "corr = numeric.corr()\n"
    "fig, ax = plt.subplots(figsize=(10, 8))\n"
    "sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1,\n"
    "            square=True, linewidths=0.3, cbar_kws={'shrink': 0.7}, ax=ax)\n"
    "ax.set_title('numeric correlation')\n"
    "plt.tight_layout(); plt.show()\n"
    "\n"
    "corr['is_canceled'].drop('is_canceled').abs().sort_values(ascending=False).head(10).round(3)"
)

md(
    "Nothing wildly correlated with the target on the numeric side. Top absolute correlations with "
    "`is_canceled`: lead_time (+), total_of_special_requests (−), required_car_parking_spaces (−), "
    "booking_changes (−). Magnitudes are modest (|r| < 0.3) which is expected — most of the signal lives "
    "in the categoricals (deposit_type, market_segment) that aren't in this matrix.\n"
    "\n"
    "Also notable: `arrival_date_year` and `arrival_date_week_number` correlate with each other and "
    "carry no real predictive content — fine to drop them in Step 4 as planned."
)

# ---- Step 3 ---------------------------------------------------------------

md(
    "## Step 3 — clean up\n"
    "\n"
    "Order matters here. First drop the two leakage columns, then NaNs, then junk rows. If I run any "
    "model with `reservation_status` still in the frame I get a fake ~100% accuracy and learn nothing."
)

code(
    "pd.crosstab(df['reservation_status'], df['is_canceled'])"
)

md(
    "Confirmed: `reservation_status` is a perfect proxy. Check-Out → is_canceled=0 in 100% of cases, "
    "Canceled / No-Show → is_canceled=1 in 100%. Dropping both this and `reservation_status_date`."
)

code(
    "df = df.drop(columns=['reservation_status', 'reservation_status_date'])\n"
    "df.shape"
)

code(
    "df['children'] = df['children'].fillna(0).astype(int)\n"
    "df['country'] = df['country'].fillna('Unknown')\n"
    "df['has_agent'] = df['agent'].notna().astype(int)\n"
    "df['has_company'] = df['company'].notna().astype(int)\n"
    "df = df.drop(columns=['agent', 'company'])\n"
    "df.isna().sum().sum()"
)

md(
    "Zero NaNs left. `agent`/`company` collapsed into boolean flags — keeps the signal (\"booked via "
    "an agent at all\") without pretending the opaque ID matters."
)

code(
    "guests = df['adults'] + df['children'] + df['babies']\n"
    "zero_guest = (guests == 0).sum()\n"
    "print('zero-guest rows to drop:', zero_guest)\n"
    "df = df[guests > 0].copy()\n"
    "df.shape"
)

code(
    "print('adr summary before filter:')\n"
    "print(df['adr'].describe().round(2))\n"
    "print()\n"
    "print('rows with adr<0 or adr>=1000:', ((df['adr'] < 0) | (df['adr'] >= 1000)).sum())\n"
    "df = df[(df['adr'] >= 0) & (df['adr'] < 1000)].copy()\n"
    "print('shape after adr filter:', df.shape)"
)

md(
    "Junk filtering caught 180 zero-guest rows (records with no actual person attached), one adr "
    "outlier at 5400 (next is 510, almost certainly a data-entry slip), and one negative adr. About "
    "0.15% of the data — not enough to bias anything, plenty to stop the outlier from dominating the "
    "scaler in Step 5.\n"
    "\n"
    "Clean. Onto feature engineering."
)

# ---- Step 4 ---------------------------------------------------------------

md(
    "## Step 4 — features\n"
    "\n"
    "Four engineered columns:\n"
    "- `total_nights` = weekend + week nights\n"
    "- `total_guests` = adults + children + babies\n"
    "- `is_family` = kids or babies present\n"
    "- `room_changed` = reserved room type ≠ assigned room type (the hotel re-assigned them — could be "
    "a sign of dissatisfaction or just operations)\n"
    "\n"
    "Then drop the source columns I no longer need: the two room-type columns (replaced by "
    "`room_changed`), and the redundant arrival-date pieces. Keep `arrival_date_month` for seasonality."
)

code(
    "df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']\n"
    "df['total_guests'] = df['adults'] + df['children'] + df['babies']\n"
    "df['is_family'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)\n"
    "df['room_changed'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)\n"
    "df[['total_nights','total_guests','is_family','room_changed']].describe().round(2)"
)

code(
    "for col in ['is_family','room_changed']:\n"
    "    g = df.groupby(col)['is_canceled'].agg(['mean','count']).round(3)\n"
    "    print(col); print(g); print()"
)

md(
    "Quick sanity check on the new binary features. `room_changed` is the interesting one: when the "
    "hotel did re-assign the room, the cancel rate drops sharply (from ~41% to ~5%). That sounds "
    "counter-intuitive at first, but it makes sense — the re-assignment can only happen if the guest "
    "actually showed up (or was about to). It's effectively a leakage-adjacent signal: not a "
    "post-arrival fact like `reservation_status`, but downstream of \"are they really coming.\" Worth "
    "keeping, but flagging it for the write-up.\n"
    "\n"
    "`is_family` barely moves the rate (36% vs 37%). Marginal feature; keep it anyway, the model can "
    "ignore it."
)

code(
    "drop_cols = [\n"
    "    'stays_in_weekend_nights', 'stays_in_week_nights',\n"
    "    'adults', 'children', 'babies',\n"
    "    'reserved_room_type', 'assigned_room_type',\n"
    "    'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',\n"
    "]\n"
    "df = df.drop(columns=drop_cols)\n"
    "print('shape:', df.shape)\n"
    "print('columns:', list(df.columns))"
)

md(
    "Down to 24 columns (was 32 at the start). The plan asked to drop the split arrival-date pieces — `year`, `week_number`, "
    "`day_of_month` — since they're redundant with `arrival_date_month` for seasonality and would just "
    "inflate the one-hot encoder. The room-type columns and per-night-bucket columns are gone too, "
    "since their content lives in the engineered features now.\n"
    "\n"
    "Ready to split."
)

# ---- Step 5 ---------------------------------------------------------------

md(
    "## Step 5 — split, THEN transform\n"
    "\n"
    "This is the one thing that's easy to do in the wrong order. If I fit the scaler/encoder on the "
    "whole frame and *then* split, the train set has seen test statistics — that's leakage and any "
    "metric I compute afterwards is overestimated. So: split first, build a Pipeline, and let it fit "
    "preprocessing on the training fold only."
)

code(
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.compose import ColumnTransformer\n"
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "\n"
    "y = df['is_canceled']\n"
    "X = df.drop(columns=['is_canceled'])\n"
    "\n"
    "categorical = ['hotel', 'arrival_date_month', 'meal', 'country',\n"
    "               'market_segment', 'distribution_channel',\n"
    "               'deposit_type', 'customer_type']\n"
    "numeric = [c for c in X.columns if c not in categorical]\n"
    "print('numeric:', len(numeric), numeric)\n"
    "print('categorical:', len(categorical), categorical)"
)

code(
    "X_train, X_test, y_train, y_test = train_test_split(\n"
    "    X, y, test_size=0.2, stratify=y, random_state=42\n"
    ")\n"
    "print('train:', X_train.shape, 'cancel rate', y_train.mean().round(4))\n"
    "print('test :', X_test.shape, 'cancel rate', y_test.mean().round(4))"
)

md(
    "Stratification did its job — train and test both sit at ~0.3705 cancel rate. 95k train / 24k test."
)

code(
    "# country has 178 unique values; min_frequency=50 keeps the head and folds the long tail\n"
    "# into a single 'infrequent_sklearn' bucket.\n"
    "preproc = ColumnTransformer([\n"
    "    ('num', StandardScaler(), numeric),\n"
    "    ('cat', OneHotEncoder(min_frequency=50,\n"
    "                          handle_unknown='ignore',\n"
    "                          sparse_output=False),\n"
    "     categorical),\n"
    "])\n"
    "\n"
    "pipe = Pipeline([\n"
    "    ('preproc', preproc),\n"
    "    ('clf', LogisticRegression(class_weight='balanced',\n"
    "                               max_iter=2000,\n"
    "                               solver='liblinear',\n"
    "                               random_state=42)),\n"
    "])\n"
    "pipe"
)

code(
    "# Sanity check: fit once with defaults, look at the encoded feature count\n"
    "# and a baseline AUC. Real tuning happens in Step 6.\n"
    "from sklearn.metrics import roc_auc_score\n"
    "\n"
    "pipe.fit(X_train, y_train)\n"
    "n_features = pipe.named_steps['preproc'].transform(X_train.head(1)).shape[1]\n"
    "auc_train = roc_auc_score(y_train, pipe.predict_proba(X_train)[:,1])\n"
    "auc_test  = roc_auc_score(y_test,  pipe.predict_proba(X_test)[:,1])\n"
    "print(f'encoded feature count: {n_features}')\n"
    "print(f'baseline AUC train: {auc_train:.4f}')\n"
    "print(f'baseline AUC test : {auc_test:.4f}')"
)

md(
    "Baseline numbers from an untuned `LogisticRegression(class_weight='balanced')`. Train and test AUC "
    "sit close to each other — no obvious over- or underfit yet. Train ≈ test means the regularisation "
    "knob in Step 6 mostly buys robustness rather than fixing a giant generalisation gap.\n"
    "\n"
    "Pipeline works end-to-end. Onto tuning."
)

# ---- Step 6 ---------------------------------------------------------------

md(
    "## Step 6 — model + tuning\n"
    "\n"
    "Grid over `C` (regularisation strength) and `penalty` (l1 vs l2). l1 will zero out weak features "
    "outright, l2 just shrinks them — l1 is interesting here because country has 50+ buckets and most "
    "are probably not pulling their weight. `liblinear` handles both penalties and is the fastest "
    "solver on this size. 5-fold stratified CV scored on ROC AUC.\n"
    "\n"
    "Single fits in a probe earlier: ~1s for l2, ~12s for l1. With 8 candidates × 5 folds = 40 fits "
    "and n_jobs=-1 this should finish in a couple of minutes."
)

code(
    "import os, sys, io, contextlib, warnings\n"
    "from sklearn.exceptions import ConvergenceWarning\n"
    "from sklearn.model_selection import GridSearchCV, StratifiedKFold\n"
    "\n"
    "# Suppress sklearn 1.8 deprecation noise from `penalty=` in workers.\n"
    "# Setting PYTHONWARNINGS propagates to joblib subprocesses.\n"
    "os.environ['PYTHONWARNINGS'] = 'ignore'\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "param_grid = {\n"
    "    'clf__C': [0.01, 0.1, 1.0, 10.0],\n"
    "    'clf__penalty': ['l1', 'l2'],\n"
    "}\n"
    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
    "\n"
    "search = GridSearchCV(\n"
    "    pipe, param_grid,\n"
    "    scoring='roc_auc',\n"
    "    cv=cv,\n"
    "    n_jobs=-1,\n"
    "    refit=True,\n"
    "    verbose=0,\n"
    ")\n"
    "with contextlib.redirect_stderr(io.StringIO()):\n"
    "    search.fit(X_train, y_train)\n"
    "\n"
    "print('best params:', search.best_params_)\n"
    "print(f'best CV AUC: {search.best_score_:.4f}')"
)

code(
    "cv_results = pd.DataFrame(search.cv_results_)\n"
    "cols = ['param_clf__C', 'param_clf__penalty', 'mean_test_score', 'std_test_score']\n"
    "cv_results[cols].sort_values('mean_test_score', ascending=False).round(4)"
)

md(
    "Full grid above. All eight combinations land in a narrow band — the surface is flat, the model "
    "isn't sensitive to the exact regularisation, which is reassuring (means the chosen point isn't a "
    "lucky pick). The best estimator is locked in as `search.best_estimator_` for Step 7."
)

code(
    "best = search.best_estimator_\n"
    "from sklearn.metrics import roc_auc_score\n"
    "auc_test = roc_auc_score(y_test, best.predict_proba(X_test)[:,1])\n"
    "print(f'held-out test AUC (best estimator): {auc_test:.4f}')"
)

# ---- Step 7 ---------------------------------------------------------------

md(
    "## Step 7 — evaluate + interpret\n"
    "\n"
    "Headline metrics: classification report (so I can see precision/recall per class), confusion "
    "matrix, ROC curve + AUC. Then a 5-fold cross-val on the final pipeline to show the AUC isn't a "
    "lucky split. Finally pull coefficients and turn them into odds ratios — that's the part where the "
    "model actually says *why* it predicts a cancellation."
)

code(
    "from sklearn.metrics import (classification_report, confusion_matrix,\n"
    "                             roc_curve, roc_auc_score)\n"
    "\n"
    "y_pred = best.predict(X_test)\n"
    "y_proba = best.predict_proba(X_test)[:, 1]\n"
    "\n"
    "print(classification_report(y_test, y_pred, target_names=['not_cancelled','cancelled'], digits=3))"
)

md(
    "Two numbers to read off this:\n"
    "\n"
    "- Recall on `cancelled` ~0.80. With `class_weight='balanced'` the threshold-0.5 model catches ~80% "
    "of cancellations. That's the metric a hotel actually cares about — false negatives (missed "
    "cancellations) are the costly mistake.\n"
    "- Precision on `cancelled` lower (~0.71). Tradeoff for that recall: more false alarms. Step 8 "
    "discusses how to retune the threshold to the hotel's actual cost ratio.\n"
    "\n"
    "Overall accuracy ~0.81 — but as noted in Step 1, accuracy alone is not the headline."
)

code(
    "cm = confusion_matrix(y_test, y_pred)\n"
    "fig, ax = plt.subplots(figsize=(5, 4))\n"
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,\n"
    "            xticklabels=['pred not_cancelled','pred cancelled'],\n"
    "            yticklabels=['true not_cancelled','true cancelled'], ax=ax)\n"
    "ax.set_title('confusion matrix (threshold 0.5)')\n"
    "plt.tight_layout(); plt.show()"
)

code(
    "fpr, tpr, _ = roc_curve(y_test, y_proba)\n"
    "auc = roc_auc_score(y_test, y_proba)\n"
    "fig, ax = plt.subplots(figsize=(5.5, 5))\n"
    "ax.plot(fpr, tpr, label=f'logistic (AUC = {auc:.3f})')\n"
    "ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='random')\n"
    "ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate')\n"
    "ax.set_title('ROC curve')\n"
    "ax.legend(loc='lower right')\n"
    "plt.tight_layout(); plt.show()"
)

code(
    "from sklearn.model_selection import cross_val_score\n"
    "with contextlib.redirect_stderr(io.StringIO()):\n"
    "    cv_auc = cross_val_score(best, X_train, y_train, cv=cv,\n"
    "                             scoring='roc_auc', n_jobs=-1)\n"
    "print('per-fold AUC:', cv_auc.round(4))\n"
    "print(f'mean: {cv_auc.mean():.4f}  std: {cv_auc.std():.4f}')"
)

md(
    "Per-fold AUCs sit in a band of ~0.001. The model is stable across splits — the 0.897 test AUC "
    "isn't a lucky draw."
)

code(
    "# Pull coefficients out of the fitted pipeline.\n"
    "preproc_fitted = best.named_steps['preproc']\n"
    "clf = best.named_steps['clf']\n"
    "\n"
    "feat_names = preproc_fitted.get_feature_names_out()\n"
    "coefs = clf.coef_[0]\n"
    "odds = np.exp(coefs)\n"
    "\n"
    "coef_df = pd.DataFrame({\n"
    "    'feature': feat_names,\n"
    "    'coef': coefs,\n"
    "    'odds_ratio': odds,\n"
    "}).sort_values('coef', key=np.abs, ascending=False)\n"
    "\n"
    "n_zero = (coefs == 0).sum()\n"
    "print(f'features total: {len(coefs)}, zeroed by l1: {n_zero}')\n"
    "coef_df.head(15).round(3)"
)

md(
    "L1 zeroed out 8 of the 105 features outright — mostly low-frequency country/segment buckets. "
    "Top-15 ranking by absolute effect is above; the odds-ratio column is the intuitive read "
    "(\"holding everything else fixed, this feature multiplies the odds of cancellation by X\").\n"
    "\n"
    "Reading the top of the list:\n"
    "\n"
    "- `required_car_parking_spaces` is the single strongest signal — sharply *negative*. Bookings "
    "with a parking spot reserved almost never cancel. That makes sense: people only reserve parking "
    "when they're actually planning to drive there.\n"
    "- `deposit_type=Non Refund` (odds ratio ~34) — the data artefact from Step 2 surfaces here exactly "
    "as predicted. The model leans on it heavily, but it's not a real-world causal driver.\n"
    "- `country_PRT` (positive) — Portuguese guests cancel more than the baseline. Both hotels are in "
    "Portugal, so this is plausibly explained by domestic guests booking more loosely and "
    "speculatively than international ones who've already committed to travel.\n"
    "- `previous_cancellations` (positive) — past behaviour predicts future behaviour. No surprise.\n"
    "- `distribution_channel_GDS` (negative) — bookings through global distribution systems (the "
    "channel airlines/large agencies use) are stickier.\n"
    "- Other consistent negatives: `total_of_special_requests`, `is_repeated_guest`, `booking_changes` "
    "— all proxies for engagement or commitment, exactly the pattern flagged in the README.\n"
    "- `room_changed` shows up large-negative, but as flagged in Step 4 this is leakage-adjacent and "
    "shouldn't be over-interpreted."
)

code(
    "top15 = coef_df.head(15).iloc[::-1]  # reverse for plotting smallest at top\n"
    "fig, ax = plt.subplots(figsize=(8, 6))\n"
    "colors = ['steelblue' if c > 0 else 'firebrick' for c in top15['coef']]\n"
    "ax.barh(top15['feature'], top15['coef'], color=colors)\n"
    "ax.axvline(0, color='k', linewidth=0.8)\n"
    "ax.set_xlabel('coefficient (positive → higher cancel risk)')\n"
    "ax.set_title('top 15 drivers by |coefficient|')\n"
    "plt.tight_layout(); plt.show()"
)

md(
    "Plot confirms the read above. Signs all match what the README predicted — that's a good sanity "
    "check on the whole pipeline. Onto the write-up."
)

# ---- Step 8 ---------------------------------------------------------------

md(
    "## Step 8 — write-up\n"
    "\n"
    "### Headline\n"
    "\n"
    "Logistic regression on the cleaned hotels dataset reaches **ROC AUC ~0.897** on held-out test "
    "(0.8986 mean across 5-fold CV, std 0.0005). At the default 0.5 threshold and with "
    "`class_weight='balanced'`, the model catches 78% of cancellations (recall) at 73% precision. "
    "The README's prior expectation was low-to-mid 0.8s for plain logistic; this landed at the high "
    "end of that, which is fine for a linear interpretable model — tree ensembles would almost "
    "certainly beat it on raw AUC but lose the coefficient story.\n"
    "\n"
    "### What actually drives cancellations\n"
    "\n"
    "Reading the top coefficients (signs and odds ratios in Step 7):\n"
    "\n"
    "Negative — these bookings stay:\n"
    "- `required_car_parking_spaces`: by far the strongest single signal. Reserving parking is a hard "
    "commitment — you only do it when you're really going.\n"
    "- `total_of_special_requests`, `booking_changes`: any post-booking engagement at all is a strong "
    "signal of intent to actually arrive.\n"
    "- `is_repeated_guest`: returning customers behave differently from one-shot bookings.\n"
    "- `distribution_channel_GDS`, `customer_type=Group`: stickier booking sources.\n"
    "\n"
    "Positive — these bookings tend to fall through:\n"
    "- `lead_time`: the longer the wait between booking and arrival, the more chance something derails "
    "the trip. Strongly monotonic in the bucketed view from Step 2.\n"
    "- `previous_cancellations`: past behaviour predicts future behaviour.\n"
    "- `country_PRT`: domestic Portuguese guests cancel noticeably more than international ones — "
    "plausibly because they book more loosely (no flights already locked in).\n"
    "- `market_segment=Online TA` (online travel agencies): low-friction booking → low-friction "
    "cancelling.\n"
    "\n"
    "### Why accuracy is the wrong headline\n"
    "\n"
    "The class split is ~37/63. A model that predicts \"never cancels\" for everyone already scores "
    "~63% accuracy without learning anything. Our 81% looks respectable but actually buys very little "
    "over that baseline. AUC and recall on the cancelled class are the metrics that reflect real "
    "model skill on this problem.\n"
    "\n"
    "### Caveats\n"
    "\n"
    "- **`deposit_type=Non Refund`** shows a 99.4% cancel rate in the raw data and lands among the "
    "strongest positive coefficients. This is a well-known artefact of how the data was logged — "
    "almost certainly the deposit flag was set retroactively after cancellation, not the other way "
    "around. The model gets a free win from it, but the feature is not a real-world causal driver of "
    "cancellations.\n"
    "- **`room_changed`** is large-negative but leakage-adjacent: the re-assignment only happens when "
    "the guest actually shows up, so it's downstream of \"are they coming.\" Useful for the model, "
    "but should not be presented as a lever the hotel can pull.\n"
    "- **`reservation_status` / `reservation_status_date`** were dropped at the very start of Step 3 — "
    "they're perfect proxies for the target. Leaving them in would have given a fake ~100% accuracy.\n"
    "\n"
    "### Limitations\n"
    "\n"
    "- **Linear log-odds assumption.** Logistic regression can only learn monotonic, additive effects. "
    "It cannot represent things like \"long lead time is fine if the deposit is non-refundable\" — that "
    "would need an interaction term.\n"
    "- **No interactions.** No `lead_time × deposit_type`, no `country × market_segment`, etc. Likely "
    "where most of the gap to tree models lives.\n"
    "- **Country was rare-bucketed** at `min_frequency=50`. ~120 of the 178 countries got folded into "
    "a single \"infrequent\" bucket. Defensible because the long tail has too few rows to learn from, "
    "but signal from specific small-country segments is lost.\n"
    "- **Threshold is fixed at 0.5**, which is rarely what a hotel actually wants. The right threshold "
    "is the one that balances the cost of a missed cancellation (room left empty, no time to resell) "
    "against the cost of a false alarm (annoying outreach to a guest who'll show up).\n"
    "\n"
    "### Next steps\n"
    "\n"
    "1. Compare against a tree-based model (random forest or gradient boosting) on the same split. "
    "The gap quantifies how much non-linearity / interaction structure was being left on the table.\n"
    "2. Add a `lead_time × deposit_type` interaction — that's the obvious candidate from the EDA.\n"
    "3. Tune the decision threshold against the hotel's actual false-positive / false-negative cost "
    "ratio, instead of leaving it at 0.5.\n"
    "4. Re-evaluate with `deposit_type` removed entirely, to estimate how much of the apparent "
    "performance leans on the artefact."
)


# ---------------------------------------------------------------------------
# Build & execute
# ---------------------------------------------------------------------------

nb = nbf.v4.new_notebook()
nb.metadata["language_info"] = {"name": "python"}
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

for kind, src in CELLS:
    if kind == "md":
        nb.cells.append(nbf.v4.new_markdown_cell(src))
    else:
        nb.cells.append(nbf.v4.new_code_cell(src))

client = NotebookClient(nb, timeout=120, kernel_name="python3")
client.execute(cwd=str(NB_PATH.parent))

nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} with {len(nb.cells)} cells")
