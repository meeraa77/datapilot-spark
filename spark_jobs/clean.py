import json
import os
import time
from pyspark.sql import functions as F
from spark_jobs.spark_session import get_spark

RAW_PATH = "data/raw/yellow_tripdata_2026-01.parquet"
OUT_PATH = "data/processed/taxi"
REPORT_PATH = "outputs/cleaning_report.json"
TARGET = "fare_amount"


def remove_invalid(df):
    """Drop rows with impossible values."""
    return df.filter(
        (F.col("fare_amount") > 0)
        & (F.col("fare_amount") < 500)
        & (F.col("trip_distance") > 0)
        & (F.col("trip_distance") < 200)
        & (F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime"))
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


def run(raw_path=RAW_PATH, out_path=OUT_PATH, report_path=REPORT_PATH):
    start = time.perf_counter()
    spark = get_spark("Clean")
    df = spark.read.parquet(raw_path)

    report = {"steps": []}

    def log(step, reason, before, after):
        report["steps"].append({
            "step": step,
            "reason": reason,
            "rows_before": before,
            "rows_after": after,
            "rows_removed": before - after,
        })
        print(f"{step}: {before} -> {after} (removed {before - after})")

    n0 = df.count()
    report["rows_raw"] = n0

    # 1. Remove exact duplicates
    df = df.dropDuplicates()
    n1 = df.count()
    log("remove_duplicates", "exact duplicate rows", n0, n1)

    # 2. Drop rows with a missing target
    df = df.dropna(subset=[TARGET])
    n2 = df.count()
    log("drop_null_target", f"{TARGET} is null", n1, n2)

    # 3. Drop impossible values
    df = remove_invalid(df)
    n3 = df.count()
    log("drop_invalid_values",
        "fare <= 0 or >= 500, distance <= 0 or >= 200, dropoff not after pickup",
        n2, n3)

    # 4. Fill missing values
    fill_cols = ["passenger_count", "RatecodeID", "congestion_surcharge", "Airport_fee"]
    df, fills = fill_nulls(df, fill_cols)
    report["fills"] = fills

    if "store_and_fwd_flag" in df.columns:
        df = df.fillna({"store_and_fwd_flag": "unknown"})

    # 5. Save cleaned Parquet
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
    run()