import json
import os
import sys
import time
from pyspark.sql import functions as F
from spark_jobs.spark_session import get_spark

DELAY_CAUSE_COLS = [
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
]
DELAY_THRESHOLD = 15  # minutes; standard "delayed" cutoff

DATASETS = {
    "taxi": {
        "raw_path": "data/raw/yellow_tripdata_2026-01.parquet",
        "out_path": "data/processed/taxi",
        "report_path": "outputs/taxi/cleaning_report.json",
        "target": "fare_amount",
    },
    "flights": {
        "raw_path": "data/raw/flights/flights.csv",
        "out_path": "data/processed/flights",
        "report_path": "outputs/flights/cleaning_report.json",
        "target": "ARRIVAL_DELAY",  # source of the derived is_delayed target
    },
}


# ---------- taxi rules (unchanged) ----------
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


# ---------- flights rules ----------
def remove_invalid_flights(df):
    """Drop rows with impossible values."""
    return df.filter(F.col("DISTANCE") > 0)


def fill_zero(df, cols):
    """Fill nulls with 0 (null means 'no delay from this cause'). Returns (df, report)."""
    cols = [c for c in cols if c in df.columns]
    counts = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols]
    ).first().asDict()
    report = {c: {"filled_with": 0, "nulls_filled": int(counts[c] or 0)} for c in cols}
    return df.fillna(0, subset=cols), report


def add_is_delayed(df, threshold=DELAY_THRESHOLD):
    """Derived target: 1 if ARRIVAL_DELAY > threshold, else 0.
    Call only after rows with null ARRIVAL_DELAY are dropped."""
    return df.withColumn("is_delayed", (F.col("ARRIVAL_DELAY") > threshold).cast("int"))


# ---------- pipeline ----------
def run(dataset="taxi", raw_path=None, out_path=None, report_path=None):
    cfg = DATASETS[dataset]
    raw_path = raw_path or cfg["raw_path"]
    out_path = out_path or cfg["out_path"]
    report_path = report_path or cfg["report_path"]
    target = cfg["target"]

    start = time.perf_counter()
    spark = get_spark("Clean")
    if raw_path.endswith(".parquet"):
        df = spark.read.parquet(raw_path)
    else:
        df = spark.read.csv(raw_path, header=True, inferSchema=True)

    report = {"dataset": dataset, "steps": []}

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
    df = df.dropna(subset=[target])
    n2 = df.count()
    reason = f"{target} is null"
    if dataset == "flights":
        reason += " (cancelled or diverted flights: no arrival outcome to predict)"
    log("drop_null_target", reason, n1, n2)

    # 3. Drop impossible values + 4. Fill missing values
    if dataset == "taxi":
        df = remove_invalid(df)
        n3 = df.count()
        log("drop_invalid_values",
            "fare <= 0 or >= 500, distance <= 0 or >= 200, dropoff not after pickup",
            n2, n3)

        fill_cols = ["passenger_count", "RatecodeID", "congestion_surcharge", "Airport_fee"]
        df, fills = fill_nulls(df, fill_cols)
        report["fills"] = fills
        if "store_and_fwd_flag" in df.columns:
            df = df.fillna({"store_and_fwd_flag": "unknown"})
    else:
        df = remove_invalid_flights(df)
        n3 = df.count()
        log("drop_invalid_values", "DISTANCE <= 0", n2, n3)

        df, fills = fill_zero(df, DELAY_CAUSE_COLS)
        report["fills"] = fills
        # Every remaining flight was not cancelled, so this is always null
        df = df.fillna({"CANCELLATION_REASON": "none"})

        df = add_is_delayed(df)
        balance = {str(r["is_delayed"]): r["count"]
                   for r in df.groupBy("is_delayed").count().collect()}
        report["derived_target"] = {
            "name": "is_delayed",
            "rule": f"ARRIVAL_DELAY > {DELAY_THRESHOLD}",
            "class_counts": balance,  # use F1, not just accuracy, if imbalanced
        }
        print("is_delayed class counts:", balance)

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
    run(sys.argv[1] if len(sys.argv) > 1 else "taxi")