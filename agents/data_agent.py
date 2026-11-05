import json
import os
from pyspark.sql import functions as F
from spark_jobs.spark_session import get_spark

NUMERIC_TYPES = ("int", "bigint", "double", "float", "smallint", "tinyint", "decimal")
TARGET_HINTS = ["target", "label", "fare_amount", "price", "churn", "default"]

def classify(name, dtype, distinct, row_count):
    if dtype.startswith("timestamp") or dtype == "date":
        return "datetime"
    if distinct > 0.95 * row_count:
        return "id"
    if dtype.startswith(NUMERIC_TYPES) and distinct >= 20:
        return "numeric"
    return "categorical"


def run(dataset_path, dataset_name="taxi", out_path="outputs/profile.json"):
    spark = get_spark("DataAgent")

    if dataset_path.endswith(".parquet"):
        df = spark.read.parquet(dataset_path)
    else:
        df = spark.read.csv(dataset_path, header=True, inferSchema=True)

    row_count = df.count()
    dup_count = row_count - df.dropDuplicates().count()

    nulls = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    ).first().asDict()

    distinct = df.select(
        [F.approx_count_distinct(c).alias(c) for c in df.columns]
    ).first().asDict()

    columns = []
    for name, dtype in df.dtypes:
        kind = classify(name, dtype, distinct[name], row_count)
        columns.append({
            "name": name,
            "type": dtype,
            "null_count": int(nulls[name] or 0),
            "distinct": int(distinct[name]),
            "kind": kind,
        })

    target = next((h for h in TARGET_HINTS if h in df.columns), None)
    problem = None
    if target:
        tkind = next(c["kind"] for c in columns if c["name"] == target)
        problem = "regression" if tkind == "numeric" else "classification"

    profile = {
        "dataset_name": dataset_name,
        "row_count": row_count,
        "column_count": len(df.columns),
        "duplicate_count": dup_count,
        "columns": columns,
        "suggested_target": target,
        "suggested_problem_type": problem,
    }

    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)

    print("Saved", out_path)
    spark.stop()
    return profile


if __name__ == "__main__":
    run("data/raw/yellow_tripdata_2026-01.parquet")