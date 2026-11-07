import json
import os
import sys
import time

from pyspark.sql import functions as F

from spark_jobs.datasets import DATASETS, read_raw
from spark_jobs.spark_session import get_spark


def remove_invalid(df):
    """Taxi: drop rows with impossible values."""
    return df.filter(
        (F.col("fare_amount") > 0)
        & (F.col("fare_amount") < 500)
        & (F.col("trip_distance") > 0)
        & (F.col("trip_distance") < 200)
        & (F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime"))
    )


def remove_invalid_flights(df):
    """Flights: keep only completed flights that have delay values."""
    return df.filter(
        (F.col("CANCELLED") == 0)
        & (F.col("DIVERTED") == 0)
        & F.col("ARRIVAL_DELAY").isNotNull()
        & F.col("DEPARTURE_DELAY").isNotNull()
    )


def fill_nulls(df, cols):
    """Fill numeric nulls with the median. Returns (df, report)."""
    fills = {}
    for c in cols:
        if c in df.columns:
            nulls = df.filter(F.col(c).isNull()).count()
            if nulls > 0:
                med = df.approxQuantile(c, [0.5], 0.01)[0]
                fills[c] = {"filled_with_median": med, "nulls_filled": nulls}
                df = df.fillna({c: med})
    return df, fills


INVALID = {"taxi": remove_invalid, "flights": remove_invalid_flights}
INVALID_REASON = {
    "taxi": "fare <= 0 or >= 500, distance <= 0 or >= 200, dropoff not after pickup",
    "flights": "cancelled or diverted flights, or missing delay values",
}


def run(name="taxi"):
    cfg = DATASETS[name]
    out_path = f"data/processed/{name}"
    report_path = f"outputs/{name}/cleaning_report.json"
    start = time.perf_counter()
    spark = get_spark("Clean")
    df = cfg["prepare"](read_raw(spark, cfg["raw"]))
    target = cfg["target"]

    report = {"dataset_name": name, "steps": []}

    def log(step, reason, before, after):
        report["steps"].append({"step": step, "reason": reason,
                                "rows_before": before, "rows_after": after,
                                "rows_removed": before - after})
        print(f"{step}: {before} -> {after} (removed {before - after})")

    n0 = df.count()
    report["rows_raw"] = n0

    df = df.dropDuplicates()
    n1 = df.count()
    log("remove_duplicates", "exact duplicate rows", n0, n1)

    df = df.dropna(subset=[target])
    n2 = df.count()
    log("drop_null_target", f"{target} is null", n1, n2)

    df = INVALID[name](df)
    n3 = df.count()
    log("drop_invalid_values", INVALID_REASON[name], n2, n3)

    df, fills = fill_nulls(df, cfg["fill_cols"])
    for c in cfg["zero_fill_cols"]:
        if c in df.columns:
            df = df.fillna({c: 0})
    report["fills"] = fills
    report["zero_filled_columns"] = cfg["zero_fill_cols"]

    for c, t in df.dtypes:
        if t == "string":
            df = df.fillna({c: "unknown"})
    df = df.drop(*[c for c in cfg["drop_cols"] if c in df.columns])
    report["dropped_columns"] = cfg["drop_cols"]

    df.write.mode("overwrite").parquet(out_path)
    final = spark.read.parquet(out_path).count()
    report["rows_final"] = final
    report["rows_removed_total"] = n0 - final
    report["seconds"] = round(time.perf_counter() - start, 1)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("Saved", out_path, "and", report_path)
    spark.stop()
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "taxi")