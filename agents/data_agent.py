"""Data Agent: prepares and profiles a dataset, writes outputs/<name>/profile.json."""
import os
import sys
import json

from pyspark.sql import functions as F


# ---------------------------------------------------------------- reading
def read_raw(spark, path):
    # KEEP YOURS: replace this body with your existing read_raw code.
    if path.endswith(".parquet"):
        return spark.read.parquet(path)
    return spark.read.csv(path, header=True, inferSchema=True)


# ---------------------------------------------------------------- prepare
def prepare_taxi(df):
    # KEEP YOURS: paste your existing prepare_taxi body here.
    return df


def prepare_flights(df):
    # KEEP YOURS: paste your existing prepare_flights body here.
    return df


# ---------------------------------------------------------------- config
# KEEP YOURS: your existing flights entry (code_cols, fill_cols, zero_fill_cols,
# drop_cols, leakage, hist_cols, scatter, hour). Only "path", "target" and
# "prepare" are new keys. Adjust the paths and targets if they are wrong.
DATASETS = {
    "taxi": {
        "path": "data/raw/yellow_tripdata_2026-01.parquet",
        "target": "fare_amount",
        "prepare": prepare_taxi,
    },
    "flights": {
        "path": "data/raw/flights.csv",
        "target": "ARRIVAL_DELAY",
        "prepare": prepare_flights,
        # KEEP YOUR EXISTING flights keys here
    },
}


# ---------------------------------------------------------------- classify
DATETIME_TYPES = ("timestamp", "timestamp_ntz", "date")
INTEGER_TYPES = ("tinyint", "smallint", "int", "bigint")
NUMERIC_TYPES = INTEGER_TYPES + ("float", "double", "decimal")

ID_RATIO = 0.95        # integer column with >= 95% unique values -> id
CATEGORY_MAX = 20      # numeric column with <= 20 distinct values -> categorical


def classify(name, dtype, distinct, rows):
    """Return 'datetime', 'id', 'numeric' or 'categorical' for one column.

    name     column name (kept for readability and future rules)
    dtype    Spark simpleString type, e.g. 'bigint', 'double', 'timestamp_ntz'
    distinct number of distinct values in the column
    rows     total number of rows
    """
    dtype = dtype.lower()
    base = dtype.split("(")[0]            # 'decimal(10,2)' -> 'decimal'

    if base in DATETIME_TYPES:
        return "datetime"

    if base in NUMERIC_TYPES:
        if base in INTEGER_TYPES and rows > 0 and distinct / rows >= ID_RATIO:
            return "id"
        if distinct <= CATEGORY_MAX:
            return "categorical"
        return "numeric"

    # string, boolean, anything else
    return "categorical"


# ---------------------------------------------------------------- profiling
def profile_dataset(df, name, target=None):
    """Profile a Spark DataFrame and return a dict (the profile.json contract)."""
    row_count = df.count()
    cols = df.columns
    types = {f.name: f.dataType.simpleString() for f in df.schema.fields}

    # Null counts and distinct counts in one pass each; no collect on full data
    null_row = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols]
    ).first().asDict()
    distinct_row = df.select(
        [F.approx_count_distinct(F.col(c)).alias(c) for c in cols]
    ).first().asDict()

    duplicate_count = row_count - df.dropDuplicates().count()

    kinds = {c: classify(c, types[c], int(distinct_row[c] or 0), row_count) for c in cols}
    numeric_cols = [c for c in cols if kinds[c] == "numeric"]
    categorical_cols = [c for c in cols if kinds[c] == "categorical"]
    datetime_cols = [c for c in cols if kinds[c] == "datetime"]
    id_cols = [c for c in cols if kinds[c] == "id"]

    # A few sample values per column (only 5 rows are collected)
    sample_rows = df.limit(5).collect()
    samples = {c: [str(r[c]) for r in sample_rows] for c in cols}

    # Problem-type rules from the guide
    warning = None
    if target and target in cols:
        target_kind = kinds[target]
        if target_kind == "numeric":
            problem_type = "regression"
        elif target_kind == "categorical":
            problem_type = "classification"
        else:
            warning = f"Target '{target}' is a {target_kind} column; falling back to clustering."
            target, problem_type = None, "clustering"
    else:
        target, problem_type = None, "clustering"

    return {
        "dataset": name,
        "row_count": row_count,
        "column_count": len(cols),
        "columns": [
            {
                "name": c,
                "type": types[c],
                "kind": kinds[c],
                "nulls": int(null_row[c] or 0),
                "distinct": int(distinct_row[c] or 0),
            }
            for c in cols
        ],
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "id_columns": id_cols,
        "duplicate_count": duplicate_count,
        "sample_values": samples,
        "suggested_target": target,
        "problem_type": problem_type,
        "warning": warning,
    }


def run(name):
    from spark_jobs.spark_session import get_spark

    if name not in DATASETS:
        raise SystemExit(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")

    cfg = DATASETS[name]
    spark = get_spark()
    try:
        df = read_raw(spark, cfg["path"])
        if cfg.get("prepare"):
            df = cfg["prepare"](df)
        profile = profile_dataset(df, name, target=cfg.get("target"))

        os.makedirs(f"outputs/{name}", exist_ok=True)
        out_path = f"outputs/{name}/profile.json"
        with open(out_path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"Done: {out_path}")
        return profile
    finally:
        spark.stop()


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "taxi"
    print(f"Profiling dataset: {dataset_name}")
    run(dataset_name)