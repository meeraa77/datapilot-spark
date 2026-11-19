import json
import os
import sys

from pyspark.sql import functions as F

from spark_jobs.datasets import DATASETS, read_raw
from spark_jobs.spark_session import get_spark

NUMERIC_TYPES = ("int", "bigint", "double", "float", "smallint", "tinyint", "decimal")


def classify(name, dtype, distinct, row_count):
    if dtype.startswith("timestamp") or dtype == "date":
        return "datetime"
    if distinct > 0.95 * row_count:
        return "id"
    if dtype.startswith(NUMERIC_TYPES) and distinct >= 20:
        return "numeric"
    return "categorical"


def run(name="taxi"):
    cfg = DATASETS[name]
    out_path = f"outputs/{name}/profile.json"
    spark = get_spark("DataAgent")
    df = cfg["prepare"](read_raw(spark, cfg["raw"]))

    row_count = df.count()
    dup_count = row_count - df.dropDuplicates().count()
    nulls = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    ).first().asDict()
    distinct = df.select(
        [F.approx_count_distinct(c).alias(c) for c in df.columns]
    ).first().asDict()

    columns = []
    for col_name, dtype in df.dtypes:
        kind = classify(col_name, dtype, distinct[col_name], row_count)
        if col_name in cfg["code_cols"] and kind != "id":
            kind = "categorical"  # category codes stored as numbers
        columns.append({
            "name": col_name, "type": dtype,
            "null_count": int(nulls[col_name] or 0),
            "distinct": int(distinct[col_name]), "kind": kind,
        })

    target = cfg["target"]
    tkind = next(c["kind"] for c in columns if c["name"] == target)
    problem = "regression" if tkind == "numeric" else "classification"

    profile = {
        "dataset_name": name,
        "row_count": row_count,
        "column_count": len(df.columns),
        "duplicate_count": dup_count,
        "columns": columns,
        "suggested_target": target,
        "suggested_problem_type": problem,
        "leakage_columns": cfg["leakage"],
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)
    print("Saved", out_path, "| problem type:", problem)
    spark.stop()
    return profile


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "taxi")