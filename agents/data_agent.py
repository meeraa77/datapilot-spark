"""Data Agent: prepares and profiles a dataset, writes outputs/<name>/profile.json."""
import os
import sys
import json

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType


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
# "prepare" are new keys. Adjust the paths and targets if mine are wrong.
DATASETS = {
    "taxi": {
        "path": "data/raw/yellow_tripdata_2026-01.parquet",
        "target": "fare_amount",          # assumed; change if needed
        "prepare": prepare_taxi,
    },
    "flights": {
        "path": "data/raw/flights.csv",   # assumed; change to your real file
        "target": "ARRIVAL_DELAY",        # assumed; change if needed
        "prepare": prepare_flights,
        # KEEP YOUR EXISTING flights keys here
    },
}


# ---------------------------------------------------------------- profiling
def profile_dataset(df, name, target=None):
    """Profile a Spark DataFrame and return a dict (the profile.json contract)."""
    row_count = df.count()
    cols = df.columns

    # Null counts in one pass, with no collect on the full data
    null_row = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols]
    ).first().asDict()

    duplicate_count = row_count - df.dropDuplicates().count()

    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    categorical_cols = [c for c in cols if c not in numeric_cols]

    # A few sample values per column (only 5 rows are collected)
    sample_rows = df.limit(5).collect()
    samples = {c: [str(r[c]) for r in sample_rows] for c in cols}

    # Problem-type rules from the guide
    if target and target in cols:
        if target in numeric_cols:
            distinct = df.select(target).distinct().count()
            problem_type = "classification" if distinct <= 20 else "regression"
        else:
            problem_type = "classification"
    else:
        target, problem_type = None, "clustering"

    return {
        "dataset": name,
        "row_count": row_count,
        "column_count": len(cols),
        "columns": [
            {"name": f.name, "type": str(f.dataType), "nulls": int(null_row[f.name] or 0)}
            for f in df.schema.fields
        ],
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "duplicate_count": duplicate_count,
        "sample_values": samples,
        "suggested_target": target,
        "problem_type": problem_type,
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