# Data Contracts

Any change to a field below must be announced to the whole team before merging, and logged in the Decisions Log at the bottom.

## 1. outputs/profile.json (Meera -> everyone)

```json
{
  "dataset_name": "taxi",
  "row_count": 3724889,
  "column_count": 20,
  "duplicate_count": 0,
  "columns": [
    {"name": "fare_amount", "type": "double", "null_count": 0, "distinct": 9000, "kind": "numeric"}
  ],
  "suggested_target": "fare_amount",
  "suggested_problem_type": "regression"
}
```

- `kind` is one of: `numeric`, `categorical`, `datetime`, `id`. It is a first guess made by simple rules.
- `suggested_problem_type` is `regression`, `classification`, or `null` (clustering).

## 2. data/processed/<dataset_name>/ (Meera -> Indrajith)

- Cleaned Parquet. Created by `python -m spark_jobs.clean`.
- Not stored in Git. Each person regenerates it locally.
- Cleaning summary: `outputs/cleaning_report.json` (rows before, after, removed, and the reason for each step).

## 3. outputs/eda_results.json (Meera -> Ardra)

Top-level keys:
`dataset_name`, `row_count`, `target`, `numeric_columns`, `categorical_columns`, `summary_stats`, `outliers`, `top_correlations`, `target_correlations`, `top_categories`, `histograms`, `by_pickup_hour`, `scatter_sample`, `seconds`

- `summary_stats.<col>`: mean, std, min, q1, median, q3, p99, max
- `outliers.<col>`: lower_bound, upper_bound, outlier_count, outlier_percent
- `top_correlations[]`: feature_a, feature_b, correlation
- `histograms.<col>[]`: bin_start, bin_end, count
- `scatter_sample[]`: at most about 3000 points (never the full data)

## 4. outputs/model_results.json (Indrajith -> Ardra)

```json
{
  "problem_type": "regression",
  "target": "fare_amount",
  "models": [
    {"name": "LinearRegression", "metrics": {"rmse": 4.2, "r2": 0.71}, "train_seconds": 12.1}
  ],
  "best_model": "LinearRegression",
  "feature_importance": [{"feature": "trip_distance", "importance": 0.61}],
  "warnings": []
}
```

## 5. outputs/insights.json (Ardra -> dashboard)

```json
{"findings": [], "limitations": [], "next_steps": []}
```

Every number in an insight must appear in the JSON files above.

## 6. agents/supervisor.py (Ardra -> all)

`run_pipeline(dataset_path)` runs all stages in order and writes every JSON file.

## Known issues (for the team)

- Do not use as features (they leak the target): `total_amount`, `tip_amount`, `tolls_amount`, `mta_tax`, `extra`, `improvement_surcharge`, `congestion_surcharge`, `Airport_fee`, `cbd_congestion_fee`.
- `VendorID`, `RatecodeID`, `payment_type`, `PULocationID`, `DOLocationID` are category codes stored as numbers.
- Some rows have odd values (for example a distance of 0.01 with a high fare). Cleaning only removes impossible values.

## Decisions Log

| Date | Decision | Who |
|---|---|---|
| 05-11-2026 | Dataset 1: NYC Yellow Taxi Jan 2026, target fare_amount (regression) | Meera |
| 05-11-2026 | Windows runs need winutils at C:\hadoop (handled in spark_session.py) | Meera |