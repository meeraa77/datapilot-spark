# ML Agent Notes

The ML Agent (`agents/ml_agent.py`) reads `outputs/<dataset>/profile.json`, picks the problem type, trains a baseline and a random forest, and writes `outputs/<dataset>/model_results.json`.

## How the problem type is chosen

| Target kind | Problem |
|---|---|
| numeric | regression |
| categorical | classification |
| none | clustering (KMeans, k = 2 to 6, silhouette) |

## Setup used for all runs

- Train/test split: 80/20, seed 42. All metrics are on the held-out 20% only.
- Training data is sampled down to about 1,000,000 rows on a laptop, and a warning records it.
- Features never include the target, ID columns, datetimes, constant columns, leakage columns from the profile, or categoricals with more than 50 values.
- Categoricals are one-hot encoded and numerics are scaled.
- Random forest: 30 trees, max depth 8.

## Taxi (regression, target: `fare_amount`)

- Extra features derived from the pickup time: `pickup_hour`, `pickup_dow`.
- Excluded as known only after the trip: `tpep_dropoff_datetime`, plus the leakage columns listed in `profile.json`.

| Model | RMSE | R² | MAE | Train time |
|---|---|---|---|---|
| LinearRegression (baseline) | 7.8282 | 0.8028 | 4.4847 | 3.0 s |
| RandomForestRegressor | 7.7772 | 0.8053 | 4.1376 | 10.7 s |

Best model: RandomForestRegressor (lowest RMSE). Total ML stage time: 43.1 s.

Observations:
- The forest beats the baseline only slightly (RMSE 7.78 vs 7.83, R² 0.805 vs 0.803). The relationship is mostly linear, so the extra complexity buys little.
- RMSE is well above MAE (7.8 vs 4.1 to 4.5), which points to a small number of large errors, likely outlier fares or distances.
- Spark logged a Cholesky warning for the linear model (correlated features) and fell back to another solver. The model still trained.
- Top features by random forest importance: `trip_distance` (0.6096), `RatecodeID_oh_1` (0.1587), `RatecodeID_oh_2` (0.0594). Trip distance dominates, as expected for fares, and the rate code (standard vs JFK fare) comes next.

## Flights (classification, target: `delayed`)

- Goal: predict delay before departure. Excluded as known after takeoff: `DEPARTURE_DELAY`, `DEPARTURE_TIME`, `TAXI_OUT`, `WHEELS_OFF`, `TAIL_NUMBER`, plus profile leakage columns (`ELAPSED_TIME`, `AIR_TIME`, `WHEELS_ON`, `TAXI_IN`, `ARRIVAL_TIME`, `ARRIVAL_DELAY`, `DIVERTED`, `CANCELLED`, `CANCELLATION_REASON`).
- Skipped for too many categories: `FLIGHT_NUMBER` (6712), `ORIGIN_AIRPORT` (602), `DESTINATION_AIRPORT` (602). `YEAR` is constant.

| Model | Accuracy | F1 | AUC | Train time |
|---|---|---|---|---|
| LogisticRegression (baseline) | 0.8213 | 0.7411 | 0.6456 | 8.4 s |
| RandomForestClassifier | 0.8214 | 0.7409 | 0.6460 | 11.0 s |

Weighted precision / recall: LogisticRegression 0.7389 / 0.8213, RandomForestClassifier 0.6747 / 0.8214.

Observations:
- Both models score about 0.821 accuracy, which equals the majority-class accuracy (0.8214). They do no better than always predicting "not delayed".
- AUC is about 0.646 for both: only modestly better than random (0.5), so the pre-departure features carry a weak signal.
- The two models are essentially tied (F1 0.7411 vs 0.7409). LogisticRegression is chosen as best by F1, and it is also faster to train.
- This is the honest result for predicting delays before takeoff, since all post-departure columns are excluded. Weather, airline-level history or time-of-day patterns would be the next features to try.

- Rows: 801,342 train and 199,909 test (about 1.0 million after sampling). Total ML stage time: 59.6 s.
- Majority-class accuracy: 0.8214. A model that always predicts the majority class would already score about 82% accuracy, so accuracy alone is not meaningful here.
- Best model (by F1): LogisticRegression (baseline). The random forest did not beat the simple baseline.
- Warnings:
  - High-cardinality columns skipped: FLIGHT_NUMBER, ORIGIN_AIRPORT, DESTINATION_AIRPORT.
  - Sampled about 1,000,000 of 5,714,008 rows for training.
  - Imbalanced target: judge by F1/AUC, not accuracy alone.

Note: modest scores are expected and honest, because everything known only after takeoff is excluded. If the target is imbalanced, judge by F1 and AUC rather than accuracy alone.

## Limitations

- Models are untuned (fixed hyperparameters) to keep runs laptop-friendly.
- High-cardinality columns such as airports are skipped, which may cost some accuracy.
- Results come from one seed and one split, with no cross-validation.