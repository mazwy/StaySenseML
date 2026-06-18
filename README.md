Hotel cancellation project — working notes

Logistic regression on the tidytuesday hotels.csv (~119k rows, two Portuguese hotels, 2015–2017). Target is is_canceled (1/0), roughly 37% positive. Goal: predict whether a booking gets cancelled before arrival, and see what actually drives it. Plan below, roughly in order. Build it as a notebook so the charts sit inline.

Step 1 — get the data in and just look at it. Load from the raw github url (or download once and read the local file, it's faster on reruns). Before touching anything: df.shape, df.info(), df.head(). Then print missing-value counts. I already expect company to be missing for basically everything (~94%), agent around 14%, country a handful, and children exactly 4 NaNs — confirm that still holds before deciding how to handle each.

Step 2 — target + basic EDA. Count is_canceled, get the rate (~37%). Note the imbalance out loud, it decides the metrics later. Then cancellation rate grouped by the categoricals that should matter: deposit_type, market_segment, hotel, customer_type. Heads up on deposit_type — Non-Refund bookings show an absurdly high cancel rate, which is backwards from intuition. It's a known artefact of how the data was logged, so flag it, don't build a story on it. For lead time: longer lead = more cancellations. Plot the lead_time distribution split by target, and also bucket lead_time and show the rate climbing across buckets. Correlation heatmap on the numerics just to eyeball, nothing rigorous.

Step 3 — clean up. Drop reservation_status and reservation_status_date first thing. These leak the target (the status column literally says "Canceled" for cancelled bookings). Leave them in and you get ~100% accuracy and a meaningless model — this is the one step you cannot skip. Then children → fillna 0, country → fillna "Unknown". Don't impute the agent/company IDs — make has_agent / has_company binary flags and drop the originals. Drop rows with zero guests (adults+children+babies == 0, they're junk records). Drop the adr outlier — there's one row around 5400, filter to adr < 1000 or so.

Step 4 — features. Build total_nights (weekend + week), total_guests, is_family (kids or babies > 0), room_changed (reserved_room_type != assigned_room_type). Drop the two room-type columns after that, plus the split-out arrival date pieces (year / week_number / day_of_month). Keep arrival_date_month as a categorical if you want seasonality.

Step 5 — split, THEN transform (order matters). y = is_canceled, X =  the rest. Split first — stratified, 80/20. Only then fit the preprocessing, and do it inside a Pipeline so the scaler/encoder fit on train only. If you scale before splitting you've leaked test stats into training. ColumnTransformer: StandardScaler on numerics, OneHotEncoder on categoricals. country is high cardinality (170+ values), so use min_frequency in the encoder to bucket the long tail, and handle_unknown="ignore" so a category that only shows up in test doesn't blow up.

Step 6 — model. LogisticRegression with class_weight="balanced" — given the 37/63 split I'd rather catch cancellations than juice raw accuracy. GridSearchCV over C (0.01 … 10) and l1/l2 penalty (liblinear or saga solver), scored on roc_auc, 5-fold stratified. Grab best_estimator_.

Step 7 — evaluate. classification_report (precision/recall/f1 per class — the cancelled class is the one I care about), confusion matrix, ROC curve + AUC. Do not report accuracy on its own: the "never cancels" baseline already scores ~63%, so accuracy looks respectable while being useless. Run cross_val on the final model too, to show it's stable and not a lucky split. Then interpret — pull coef_, exp() them into odds ratios, plot the top 15 by absolute value. I'd expect lead_time, previous_cancellations, deposit_type, market_segment to land positive; special_requests, required_car_parking_spaces, repeated_guest negative.

Step 8 — write-up. Discussion covers: what the AUC actually came out to (probably low-to-mid 0.8s for plain logistic — fine for a linear interpretable model, ensembles would beat it), what drives cancellations from the coefficients, why accuracy is the wrong headline here, the deposit_type caveat, and limitations (linear log-odds assumption, no interactions yet, country collapsed via rare-bucketing). Next steps: compare against a tree / random forest, add a lead_time × deposit_type interaction, and tune the decision threshold to the hotel's real false-positive vs false-negative cost rather than leaving it at 0.5.

Things I keep forgetting / don't screw up:


drop the leakage columns before anything else
split before scaling, always
the children NaN is only 4 rows, easy to miss
deposit_type Non-Refund is weird — caveat it, don't over-read it
lead with AUC + recall, accuracy alone is misleading at this base rate

Content