# ML Notes (Indrajith)

## Problem-type rules
- numeric target -> regression
- categorical target -> classification
- no target -> clustering (KMeans, k = 2..6, best silhouette)

## Datasets
| Dataset | Target | Problem | Baseline | Improved | Main metrics |
|---|---|---|---|---|---|
| taxi | fare_amount | regression | LinearRegression | RandomForestRegressor | RMSE, R2, MAE |
| flights | delayed (arrival delay > 15 min) | classification | LogisticRegression | RandomForestClassifier | F1, AUC, accuracy |

## Leakage decisions
- Taxi: total_amount, tip_amount, tolls_amount, taxes and fees are excluded because they add up to the fare.
- Taxi: dropoff time is excluded because it is only known after the trip.
- Flights: the model predicts BEFORE departure, so DEPARTURE_DELAY, DEPARTURE_TIME, TAXI_OUT, WHEELS_OFF and everything after takeoff are excluded.
- ID columns and the target are never used as features (tested in tests/test_ml.py).

## Evaluation
- 80/20 random split with seed 42; metrics are computed on the 20% test part only.
- Training data is sampled to about 1,000,000 rows (laptop limit).
- Flights is imbalanced, so F1 and AUC are reported, not accuracy alone.

## Results (fill in from model_results.json)
| Dataset | Model | Metrics | Train seconds |
|---|---|---|---|
| taxi | | | |
| flights | | | |

## Limitations
- Columns with more than 50 categories (airports, zones) are skipped. Target or frequency encoding would be a next step.
- No cross-validation, because it is slow on a laptop. A single split was used.
- Flights accuracy is modest on purpose: the strongest predictors are only known after takeoff.