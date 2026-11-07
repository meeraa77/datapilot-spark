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
| columns | list | One object per column: `name`, `type`, `kind`, `nulls`, `distinct` |
| numeric_columns | list[string] | Columns with kind `numeric` |
| categorical_columns | list[string] | Columns with kind `categorical` (strings, booleans, numeric columns with 20 or fewer distinct values) |
| datetime_columns | list[string] | Columns with kind `datetime` (timestamp or date types) |
| id_columns | list[string] | Columns with kind `id` (integer columns that are 95% or more unique). **Never use as ML features.** |
| duplicate_count | int | Rows that are exact duplicates |
| sample_values | object | Column name -> list of 5 sample values (strings) |
| suggested_target | string or null | Suggested target column; null means no target |
| problem_type | string | `classification`, `regression` or `clustering` |
| warning | string or null | Explains why the problem type fell back to clustering; otherwise null |

Column `kind` is one of `numeric`, `categorical`, `datetime`, `id`.
Every column appears in exactly one of the four lists above.
`type` is the Spark type as text (for example `bigint`, `double`, `timestamp_ntz`).
`distinct` is an approximate distinct count.

Example (taxi):

```json
{
  "dataset": "taxi",
  "row_count": 3724889,
  "column_count": 20,
  "columns": [
    {"name": "RatecodeID", "type": "bigint", "kind": "categorical", "nulls": 1088058, "distinct": 7},
    {"name": "fare_amount", "type": "double", "kind": "numeric", "nulls": 0, "distinct": 9000},
    {"name": "tpep_pickup_datetime", "type": "timestamp_ntz", "kind": "datetime", "nulls": 0, "distinct": 3400000}
  ],
  "numeric_columns": ["fare_amount"],
  "categorical_columns": ["RatecodeID", "store_and_fwd_flag"],
  "datetime_columns": ["tpep_pickup_datetime", "tpep_dropoff_datetime"],
  "id_columns": [],
  "duplicate_count": 0,
  "sample_values": {"fare_amount": ["9.3", "7.9", "14.2", "6.5", "11.0"]},
  "suggested_target": "fare_amount",
  "problem_type": "regression",
  "warning": null
}
```

Rules:
- Column kind: timestamp or date type -> datetime; integer type with 95% or
  more unique values -> id; other numeric type with 20 or fewer distinct
  values -> categorical; other numeric type -> numeric; everything else
  (string, boolean) -> categorical.
- Problem type follows the target's kind: categorical target ->
  classification; numeric target -> regression; no target, or a target that
  is an id or datetime column -> clustering (with a `warning` in the last case).

---

## 2. Cleaned data  (Meera -> Indrajith)

- Location: `data/processed/<dataset>/` (Parquet)
- Column lists come from `profile.json` (`numeric_columns`,
  `categorical_columns`, `datetime_columns`, `id_columns`)
- `id_columns` must not be used as features. Datetime columns are not used
  directly as features; derive parts (hour, weekday) first if needed.
- Rows changed or dropped are reported in `outputs/<dataset>/cleaning_report.json`

---

## 3. eda_results.json  (Meera -> Ardra)

Written by `agents/eda_agent.py`.
Run: `python -m agents.eda_agent <dataset>`
Needs `outputs/<dataset>/profile.json` and `data/processed/<dataset>/`.

| Field | Type | Meaning |
|---|---|---|
| dataset | string | Dataset name |
| row_count | int | Rows in the cleaned data |
| target | string | Target column |
| summary_stats | object | Column -> `mean`, `stddev`, `min`, `max` (numeric columns, plus the target if numeric) |
| top_correlations | list | Up to 10 numeric columns, strongest first by absolute value. Each: `col_a` (feature), `col_b` (target), `correlation`, `leakage`. Empty if the target is not numeric. |
| top_categories | object | Column -> up to 5 `{value, count}` items (value is a string). Up to 8 columns; the target comes first if categorical. |
| outliers | object | Numeric column -> `count`, `method` ("IQR"), `lower`, `upper` (the IQR bounds) |
| chart_data.histograms | object | Column -> `bins` (20 left edges), `counts` (20 values), `clipped_to` ("p1-p99") |
| chart_data.scatter | object or null | `x`, `y`, `points` (up to 500 `[x, y]` pairs) |
| chart_data.trend_by_hour | object or null | `target`, `hours`, `counts`, `avg_target` (lists of equal length) |
| seconds | float | Time the EDA run took |

Rules:
- `leakage: true` means the column is computed from or recorded after the
  target (list in `spark_jobs/datasets.py`). Such correlations must **not** be
  presented as findings or drivers.
- Histograms are clipped to the 1st-99th percentile, so they do not show
  extreme outliers. Outlier counts cover all rows.
- Outlier bounds and percentiles are approximate (Spark approxQuantile, error 0.001).
- Correlations are Pearson. For a 0/1 target such as flights `delayed`, this is a
  point-biserial correlation.

Example (taxi, shortened):

```json
{
  "dataset": "taxi",
  "row_count": 3516799,
  "target": "fare_amount",
  "summary_stats": {
    "fare_amount": {"mean": 21.0817, "stddev": 17.336, "min": 0.01, "max": 498.0}
  },
  "top_correlations": [
    {"col_a": "total_amount", "col_b": "fare_amount", "correlation": 0.9659, "leakage": true},
    {"col_a": "trip_distance", "col_b": "fare_amount", "correlation": 0.8595, "leakage": false}
  ],
  "top_categories": {
    "payment_type": [{"value": "1", "count": 0}]
  },
  "outliers": {
    "fare_amount": {"count": 0, "method": "IQR", "lower": 0.0, "upper": 0.0}
  },
  "chart_data": {
    "histograms": {"fare_amount": {"bins": [0.0], "counts": [0], "clipped_to": "p1-p99"}},
    "scatter": {"x": "trip_distance", "y": "fare_amount", "points": [[1.2, 7.9]]},
    "trend_by_hour": {"target": "fare_amount", "hours": [0, 1], "counts": [0, 0], "avg_target": [0.0, 0.0]}
  },
  "seconds": 36.9
}
```

(`correlation`, `mean`, `stddev`, `min` and `max` above are real taxi values.
Other numbers are placeholders to show the shape.)

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
| 07-11-2026 | profile.json: added `datetime_columns`, `id_columns`, `warning`, and per-column `kind` and `distinct`. Datetime columns no longer appear under `categorical_columns`. Problem type now follows the target's kind. | Indrajith, Ardra |
| 07-11-2026 | eda_results.json is final (no longer DRAFT). New fields vs the draft: row_count, target, leakage flag in top_correlations, lower/upper in outliers, clipped_to in histograms, trend_by_hour, seconds. | Indrajith, Ardra |