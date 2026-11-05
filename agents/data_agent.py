import json
import os
import sys
from pyspark import StorageLevel
from pyspark.sql import functions as F
from spark_jobs.spark_session import get_spark

NUMERIC_TYPES = ("int", "bigint", "double", "float", "smallint", "tinyint", "decimal")
TARGET_HINTS = ["target", "label", "fare_amount", "price", "churn", "default"]

# Per-dataset rules. Taxi needs none (hint matching works).
DATASET_CONFIG = {
    "flights": {
        "target": "is_delayed",
        "problem_type": "classification",
        "derived_target": {
            "name": "is_delayed",
            "source_column": "ARRIVAL_DELAY",
            "rule": "ARRIVAL_DELAY > 15",
            "created_in": "spark_jobs/clean.py",
        },
        "id_like": ["FLIGHT_NUMBER", "TAIL_NUMBER"],
        # Assumption: we predict delay at scheduling time, so anything
        # only known after departure/arrival is leakage.
        "leakage": [
            "DEPARTURE_TIME", "DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF",
            "ELAPSED_TIME", "AIR_TIME", "WHEELS_ON", "TAXI_IN",
            "ARRIVAL_TIME", "ARRIVAL_DELAY", "DIVERTED", "CANCELLED",
            "CANCELLATION_REASON", "AIR_SYSTEM_DELAY", "SECURITY_DELAY",
            "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
        ],
    },
}


def classify(name, dtype, distinct, row_count):
    if dtype.startswith("timestamp") or dtype == "date":
        return "datetime"
    if distinct > 0.95 * row_count:
        return "id"
    if dtype.startswith(NUMERIC_TYPES) and distinct >= 20:
        return "numeric"
    return "categorical"


def assign_role(name, kind, cfg, target):
    if name == target:
        return "target"
    if name in cfg.get("leakage", []):
        return "leakage"
    if kind == "id" or name in cfg.get("id_like", []):
        return "id"
    return "feature"


def run(dataset_path, dataset_name="taxi", out_path=None):
    if out_path is None:
        out_path = f"outputs/{dataset_name}/profile.json"
    cfg = DATASET_CONFIG.get(dataset_name, {})
    spark = get_spark("DataAgent")

    if dataset_path.endswith(".parquet"):
        df = spark.read.parquet(dataset_path)
    else:
        df = spark.read.csv(dataset_path, header=True, inferSchema=True)

    # Several passes follow; keep the data in memory/disk instead of rereading the CSV
    df = df.persist(StorageLevel.MEMORY_AND_DISK)

    row_count = df.count()
    dup_count = row_count - df.dropDuplicates().count()

    nulls = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    ).first().asDict()

    distinct = df.select(
        [F.approx_count_distinct(c).alias(c) for c in df.columns]
    ).first().asDict()

    # Target: explicit config first, then name hints
    target = cfg.get("target") or next((h for h in TARGET_HINTS if h in df.columns), None)

    columns = []
    for name, dtype in df.dtypes:
        kind = classify(name, dtype, distinct[name], row_count)
        columns.append({
            "name": name,
            "type": dtype,
            "null_count": int(nulls[name] or 0),
            "distinct": int(distinct[name]),
            "kind": kind,
            "role": assign_role(name, kind, cfg, target),
        })

    problem = cfg.get("problem_type")
    if problem is None and target:
        tkind = next((c["kind"] for c in columns if c["name"] == target), None)
        if tkind:
            problem = "regression" if tkind == "numeric" else "classification"

    profile = {
        "dataset_name": dataset_name,
        "row_count": row_count,
        "column_count": len(df.columns),
        "duplicate_count": dup_count,
        "columns": columns,
        "suggested_target": target,
        "suggested_problem_type": problem,
        "derived_target": cfg.get("derived_target"),
        "excluded_columns": [c["name"] for c in columns if c["role"] in ("leakage", "id")],
    }

    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)

    print("Saved", out_path)
    df.unpersist()
    spark.stop()
    return profile


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/yellow_tripdata_2026-01.parquet"
    name = sys.argv[2] if len(sys.argv) > 2 else "taxi"
    run(path, name)