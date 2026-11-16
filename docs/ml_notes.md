# ML Agent notes

## Problem-type rules
- Numeric target in profile.json -> regression
- Categorical target -> classification
- No target -> clustering (KMeans, k=2 to 6, best by silhouette)

## Models tried
| Dataset | Target | Baseline | Improved | Metrics |
|---|---|---|---|---|
| taxi | fare_amount | LinearRegression | RandomForestRegressor | RMSE, R2, MAE |
| flights | delayed | LogisticRegression | RandomForestClassifier | accuracy, F1, AUC |

Trained on 80% of the data, tested on the held-out 20% (seed 42).

## Leakage decisions
- Columns in leakage_columns (profile.json), the target, ID columns, constant columns and raw datetimes are excluded.
- taxi: tpep_dropoff_datetime is excluded (known only after the trip).
- flights: the model predicts BEFORE departure, so DEPARTURE_DELAY, DEPARTURE_TIME, TAXI_OUT, WHEELS_OFF and TAIL_NUMBER are excluded.

## Sampling
- Data over 1,000,000 rows is sampled to about 1,000,000 for training. A warning is saved in model_results.json.

## Limitations
- Columns with more than 50 categories (airports, taxi zones) are skipped. Later improvement: target or frequency encoding.
- Flights is imbalanced, so judge by F1 and AUC. A modest AUC is expected because post-takeoff information is removed.
- Random forests use 30 trees, depth 8, with no tuning.