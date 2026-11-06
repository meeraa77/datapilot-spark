# Data Contracts

Any change to a field name or structure must be announced in the group chat
and logged in `docs/decisions_log.md`.

All outputs are written per dataset: `outputs/<dataset>/<file>.json`
(datasets so far: `taxi`, `flights`).

---

## 1. profile.json  (Meera -> everyone)

Written by `agents/data_agent.py`.
Run: `python -m agents.data_agent <dataset>`

| Field | Type | Meaning |
|---|---|---|
| dataset | string | Dataset name |
| row_count | int | Number of rows |
| column_count | int | Number of columns |
| columns | list | One object per column: `name`, `type`, `nulls` |
| numeric_columns | list[string] | Columns with numeric types |
| categorical_columns | list[string] | All other columns |
| duplicate_count | int | Rows that are exact duplicates |
| sample_values | object | Column name -> list of 5 sample values (strings) |
| suggested_target | string or null | Suggested target column; null means no target |
| problem_type | string | `classification`, `regression` or `clustering` |

Example (taxi):

```json
{
  "dataset": "taxi",
  "row_count": 3724889,
  "column_count": 20,
  "columns": [
    {"name": "RatecodeID", "type": "LongType()", "nulls": 1088058},
    {"name": "fare_amount", "type": "DoubleType()", "nulls": 0}
  ],
  "numeric_columns": ["RatecodeID", "fare_amount"],
  "categorical_columns": ["store_and_fwd_flag"],
  "duplicate_count": 0,
  "sample_values": {"fare_amount": ["9.3", "7.9", "14.2", "6.5", "11.0"]},
  "suggested_target": "fare_amount",
  "problem_type": "regression"
}
```

Rules:
- Problem type: categorical target -> classification; numeric target with
  20 or fewer distinct values -> classification; other numeric target ->
  regression; no target -> clustering.

---

## 2. Cleaned data  (Meera -> Indrajith)

- Location: `data/processed/<dataset>/` (Parquet)
- Column lists come from `profile.json` (`numeric_columns`, `categorical_columns`)
- Rows changed or dropped are reported in `outputs/<dataset>/cleaning_report.json`

---

## 3. eda_results.json  (Meera -> Ardra)   DRAFT

```json
{
  "dataset": "taxi",
  "summary_stats": {
    "fare_amount": {"mean": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0}
  },
  "top_correlations": [
    {"col_a": "trip_distance", "col_b": "fare_amount", "correlation": 0.0}
  ],
  "top_categories": {
    "payment_type": [{"value": "1", "count": 0}]
  },
  "outliers": {
    "fare_amount": {"count": 0, "method": "IQR"}
  },
  "chart_data": {
    "histograms": {"fare_amount": {"bins": [], "counts": []}},
    "scatter": {"x": "trip_distance", "y": "fare_amount", "points": []}
  }
}
```

---

## 4. model_results.json  (Indrajith -> Ardra)   DRAFT

```json
{
  "dataset": "taxi",
  "problem_type": "regression",
  "target": "fare_amount",
  "model_name": "GBTRegressor",
  "baseline_model": "LinearRegression",
  "metrics": {"rmse": 0.0, "r2": 0.0},
  "baseline_metrics": {"rmse": 0.0, "r2": 0.0},
  "feature_importance": [{"feature": "trip_distance", "importance": 0.0}],
  "training_time_sec": 0.0,
  "warnings": []
}
```

---

## 5. insights.json  (Ardra -> dashboard)   DRAFT

```json
{
  "dataset": "taxi",
  "findings": ["..."],
  "limitations": ["..."],
  "next_steps": ["..."]
}
```

Rule: insights may only use numbers that appear in the other JSON files.

---

## 6. Supervisor interface  (Ardra -> all)

`agents/supervisor.py` provides `run_pipeline(dataset_name)`.
It runs all stages in order and writes every JSON file above.

---

## Change log

| Date | Change | Announced to |
|---|---|---|
| 06-11-2026 | Initial draft of all contracts | Indrajith, Ardra |