"""EDA Agent: explores the cleaned data and writes outputs/<name>/eda_results.json.

Run:  python -m agents.eda_agent <dataset>

Needs:
  outputs/<name>/profile.json   (python -m agents.data_agent <name>)
  data/processed/<name>/        (python -m spark_jobs.clean <name>)

All heavy work stays in Spark. Only small results are collected
(a few numbers per column, 20 histogram bins, 500 scatter points).
"""
import json
import os
import sys
import time

from pyspark.sql import functions as F

from spark_jobs.datasets import DATASETS
from spark_jobs.spark_session import get_spark

TOP_CORR = 10            # correlations kept in the output
TOP_CATS = 5             # values kept per categorical column
MAX_CAT_COLUMNS = 8      # categorical columns shown
MAX_CAT_DISTINCT = 1000  # skip categorical columns with more distinct values
HIST_BINS = 20
QUANTILE_ERROR = 0.001   # approxQuantile accuracy (smaller = more exact, a bit slower)
SCATTER_POINTS = 500

NUMERIC_DTYPES = ("tinyint", "smallint", "int", "bigint", "float", "double")


# ---------------------------------------------------------------- helpers
def num(x, nd=4):
    """Make a value JSON-safe: None for null/NaN/inf, otherwise a rounded float."""
    if x is None:
        return None
    x = float(x)
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return round(x, nd)


def is_numeric_dtype(dtype):
    return dtype in NUMERIC_DTYPES or dtype.startswith("decimal")


def load_profile(name):
    path = f"outputs/{name}/profile.json"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run: python -m agents.data_agent {name}")
    with open(path) as f:
        return json.load(f)


def pick_columns(profile, cfg, df_cols):
    """Choose numeric and categorical columns for EDA from profile.json.

    Columns listed in cfg['code_cols'] are codes (for example PULocationID),
    so they are treated as categorical even if they are stored as numbers.
    """
    code = [c for c in cfg.get("code_cols", []) if c in df_cols]
    distinct = {c["name"]: c["distinct"] for c in profile["columns"]}

    numeric = [c for c in profile["numeric_columns"]
               if c in df_cols and c not in code]

    categorical = list(dict.fromkeys(
        [c for c in profile["categorical_columns"] if c in df_cols] + code))
    categorical = [c for c in categorical
                   if 2 <= distinct.get(c, 2) <= MAX_CAT_DISTINCT]
    categorical.sort(key=lambda c: distinct.get(c, 0))
    categorical = categorical[:MAX_CAT_COLUMNS]

    target = cfg["target"]
    if target in categorical:                      # always show the target first
        categorical.remove(target)
        categorical.insert(0, target)
    return numeric, categorical


# ---------------------------------------------------------------- analyses
def summary_stats(df, cols):
    """mean, stddev, min, max for each column, computed in one Spark pass."""
    aggs = []
    for c in cols:
        col = F.col(c)
        aggs += [F.mean(col).alias(f"{c}__mean"), F.stddev(col).alias(f"{c}__std"),
                 F.min(col).alias(f"{c}__min"), F.max(col).alias(f"{c}__max")]
    row = df.agg(*aggs).first().asDict()
    return {c: {"mean": num(row[f"{c}__mean"]), "stddev": num(row[f"{c}__std"]),
                "min": num(row[f"{c}__min"]), "max": num(row[f"{c}__max"])}
            for c in cols}


def top_correlations(df, cols, target, leakage):
    """Correlation of each numeric column with the target, in one Spark pass.

    Columns listed in cfg['leakage'] are flagged, so nobody reads
    'total_amount is linked to fare_amount' as a real finding.
    """
    if not is_numeric_dtype(dict(df.dtypes)[target]):
        return []
    cands = [c for c in cols if c != target]
    if not cands:
        return []
    row = df.agg(*[F.corr(F.col(c), F.col(target)).alias(c) for c in cands]).first().asDict()
    items = [(c, num(row[c])) for c in cands if num(row[c]) is not None]
    items.sort(key=lambda t: abs(t[1]), reverse=True)
    return [{"col_a": c, "col_b": target, "correlation": v, "leakage": c in leakage}
            for c, v in items[:TOP_CORR]]


def top_categories(df, cols):
    out = {}
    for c in cols:
        rows = (df.groupBy(c).count().orderBy(F.desc("count")).limit(TOP_CATS).collect())
        out[c] = [{"value": str(r[c]), "count": int(r["count"])} for r in rows]
    return out


def outliers(df, cols):
    """IQR rule: below Q1 - 1.5*IQR or above Q3 + 1.5*IQR."""
    bounds = {}
    for c in cols:
        q = df.approxQuantile(c, [0.25, 0.75], QUANTILE_ERROR)
        if len(q) == 2:
            iqr = q[1] - q[0]
            bounds[c] = (q[0] - 1.5 * iqr, q[1] + 1.5 * iqr)
    if not bounds:
        return {}
    row = df.agg(*[
        F.sum(F.when((F.col(c) < lo) | (F.col(c) > hi), 1).otherwise(0)).alias(c)
        for c, (lo, hi) in bounds.items()
    ]).first().asDict()
    return {c: {"count": int(row[c] or 0), "method": "IQR",
                "lower": num(lo), "upper": num(hi)}
            for c, (lo, hi) in bounds.items()}


def histograms(df, cols):
    """Histogram per column, clipped to the 1st-99th percentile so that
    outliers do not squash the chart. 'bins' are the left edges."""
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        q = df.approxQuantile(c, [0.01, 0.99], QUANTILE_ERROR)
        if len(q) < 2 or q[0] == q[1]:
            continue
        lo, hi = q
        width = (hi - lo) / HIST_BINS
        rows = (df.filter((F.col(c) >= lo) & (F.col(c) <= hi))
                  .select(F.least(F.floor((F.col(c) - lo) / width),
                                  F.lit(HIST_BINS - 1)).cast("int").alias("b"))
                  .groupBy("b").count().collect())
        counts = [0] * HIST_BINS
        for r in rows:
            counts[r["b"]] = int(r["count"])
        out[c] = {"bins": [num(lo + i * width) for i in range(HIST_BINS)],
                  "counts": counts, "clipped_to": "p1-p99"}
    return out


def scatter(df, x, y, n_rows):
    """A small random sample of (x, y) pairs for a scatter plot."""
    frac = min(1.0, SCATTER_POINTS * 4 / max(n_rows, 1))
    rows = df.select(x, y).sample(False, frac, seed=42).limit(SCATTER_POINTS).collect()
    return {"x": x, "y": y, "points": [[num(r[x]), num(r[y])] for r in rows]}


def trend_by_hour(df, hour_expr, target):
    """Row count and average target for each hour of the day."""
    rows = (df.select(hour_expr.alias("hour"), F.col(target).cast("double").alias("t"))
              .groupBy("hour")
              .agg(F.count("*").alias("n"), F.avg("t").alias("avg_target"))
              .orderBy("hour").collect())
    rows = [r for r in rows if r["hour"] is not None]
    return {"target": target,
            "hours": [int(r["hour"]) for r in rows],
            "counts": [int(r["n"]) for r in rows],
            "avg_target": [num(r["avg_target"]) for r in rows]}


# ---------------------------------------------------------------- run
def run(name):
    if name not in DATASETS:
        raise SystemExit(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")

    cfg = DATASETS[name]
    target = cfg["target"]
    profile = load_profile(name)

    data_path = f"data/processed/{name}"
    if not os.path.exists(data_path):
        raise SystemExit(f"{data_path} not found. Run: python -m spark_jobs.clean {name}")

    start = time.perf_counter()
    spark = get_spark("EDA")
    try:
        df = spark.read.parquet(data_path).cache()
        n_rows = df.count()
        dtypes = dict(df.dtypes)
        if target not in dtypes:
            raise SystemExit(f"Target '{target}' is not in the cleaned data.")

        numeric, categorical = pick_columns(profile, cfg, df.columns)
        stat_cols = list(numeric)
        if target not in stat_cols and is_numeric_dtype(dtypes[target]):
            stat_cols.append(target)         # e.g. mean of 'delayed' = delay rate
        leakage = set(cfg.get("leakage", []))

        print(f"{name}: {n_rows} rows, {len(stat_cols)} numeric, "
              f"{len(categorical)} categorical columns")

        chart_data = {
            "histograms": histograms(df, cfg.get("hist_cols", [])),
            "scatter": None,
            "trend_by_hour": None,
        }
        sx, sy = cfg.get("scatter", [None, None])
        if sx in df.columns and sy in df.columns:
            chart_data["scatter"] = scatter(df, sx, sy, n_rows)
        if cfg.get("hour"):
            chart_data["trend_by_hour"] = trend_by_hour(df, cfg["hour"](), target)

        results = {
            "dataset": name,
            "row_count": n_rows,
            "target": target,
            "summary_stats": summary_stats(df, stat_cols),
            "top_correlations": top_correlations(df, stat_cols, target, leakage),
            "top_categories": top_categories(df, categorical),
            "outliers": outliers(df, numeric),
            "chart_data": chart_data,
            "seconds": round(time.perf_counter() - start, 1),
        }

        os.makedirs(f"outputs/{name}", exist_ok=True)
        out_path = f"outputs/{name}/eda_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Done: {out_path}")
        return results
    finally:
        spark.stop()


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "taxi"
    print(f"Running EDA for: {dataset_name}")
    run(dataset_name)