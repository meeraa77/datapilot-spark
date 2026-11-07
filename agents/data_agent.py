"""Data Agent: prepares and profiles a dataset, writes outputs/<name>/profile.json."""
import os
import sys
import json

from pyspark.sql import functions as F


# ---------------------------------------------------------------- reading
def read_raw(spark, path):
    if path.endswith(".csv"):
        return spark.read.csv(path, header=True, inferSchema=True)
    return spark.read.parquet(path)


# ---------------------------------------------------------------- prepare
def prepare_taxi(df):
    return df


def prepare_flights(df):
    """Target: 1 = arrived more than 15 minutes late, else 0."""
    return df.withColumn(
        "delayed",
        F.when(F.col("ARRIVAL_DELAY").isNull(), F.lit(None).cast("int"))
         .when(F.col("ARRIVAL_DELAY") > 15, F.lit(1))
         .otherwise(F.lit(0)))


# ---------------------------------------------------------------- config
DATASETS = {
    "taxi": {
        "raw": "data/raw/yellow_tripdata_2026-01.parquet",
        "prepare": prepare_taxi,
        "target": "fare_amount",
        "code_cols": ["VendorID", "RatecodeID", "payment_type",
                      "PULocationID", "DOLocationID"],
        "fill_cols": ["passenger_count", "RatecodeID",
                      "congestion_surcharge", "Airport_fee"],
        "zero_fill_cols": [],
        "drop_cols": [],
        "leakage": ["total_amount", "tip_amount", "tolls_amount", "mta_tax",
                    "extra", "improvement_surcharge", "congestion_surcharge",
                    "Airport_fee", "cbd_congestion_fee"],
        "hist_cols": ["fare_amount", "trip_distance"],
        "scatter": ["trip_distance", "fare_amount"],
        "hour": lambda: F.hour("tpep_pickup_datetime"),
    },
    "flights": {
        "raw": "data/raw/flights.parquet",
        "prepare": prepare_flights,
        "target": "delayed",
        "code_cols": ["FLIGHT_NUMBER", "DAY", "MONTH", "DAY_OF_WEEK"],
        "fill_cols": ["DEPARTURE_TIME", "TAXI_OUT", "WHEELS_OFF",
                      "SCHEDULED_TIME", "ELAPSED_TIME", "AIR_TIME",
                      "WHEELS_ON", "TAXI_IN", "ARRIVAL_TIME"],
        "zero_fill_cols": ["AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
                           "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"],
        "drop_cols": ["CANCELLATION_REASON"],
        "leakage": ["ARRIVAL_DELAY", "ARRIVAL_TIME", "WHEELS_ON", "TAXI_IN",
                    "ELAPSED_TIME", "AIR_TIME", "AIR_SYSTEM_DELAY",
                    "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY",
                    "WEATHER_DELAY", "CANCELLED", "DIVERTED",
                    "CANCELLATION_REASON"],
        "hist_cols": ["ARRIVAL_DELAY", "DISTANCE"],
        "scatter": ["DISTANCE", "ARRIVAL_DELAY"],
        "hour": lambda: F.floor(F.col("SCHEDULED_DEPARTURE") / 100).cast("int"),
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
        df = read_raw(spark, cfg["raw"])
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