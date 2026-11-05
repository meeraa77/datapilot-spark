import os
import platform
import time

import pandas as pd
from pyspark.sql import functions as F

from spark_jobs.clean import fill_nulls, remove_invalid
from spark_jobs.spark_session import get_spark

RAW = "data/raw/yellow_tripdata_2026-01.parquet"
BENCH_DIR = "data/benchmark"
RESULTS_CSV = "outputs/benchmark_results.csv"
SUMMARY_CSV = "outputs/benchmark_summary.csv"
CHART_PNG = "outputs/scalability.png"

# 0.25x = a quarter of the file, 2x = the file stacked twice.
# Add "4x": 4 only if your laptop has 16 GB RAM or more.
SCALES = {"0.25x": 0.25, "1x": 1, "2x": 2}
REPEATS = 3
PANDAS_MAX_SCALE = 2
FILL_COLS = ["passenger_count", "RatecodeID", "congestion_surcharge", "Airport_fee"]


def make_dataset(spark, label, factor):
    path = f"{BENCH_DIR}/size_{label}"
    if os.path.exists(path):
        return path
    base = spark.read.parquet(RAW)
    if factor < 1:
        df = base.sample(fraction=factor, seed=42).withColumn("copy_id", F.lit(0))
    else:
        df = None
        for i in range(int(factor)):
            part = base.withColumn("copy_id", F.lit(i))  # keeps copies distinct
            df = part if df is None else df.unionByName(part)
    df.write.mode("overwrite").parquet(path)
    return path


def folder_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return round(total / 1e6, 1)


def stage_profile(spark, path):
    df = spark.read.parquet(path)
    df.count()
    df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).first()
    df.select([F.approx_count_distinct(c).alias(c) for c in df.columns]).first()


def stage_clean(spark, path):
    df = spark.read.parquet(path).dropDuplicates()
    df = remove_invalid(df)
    df, _ = fill_nulls(df, FILL_COLS)
    df.write.mode("overwrite").parquet(f"{BENCH_DIR}/_clean_tmp")


def stage_eda(spark, path):
    df = spark.read.parquet(path)
    df.agg(F.mean("fare_amount"), F.stddev("fare_amount"),
           F.corr("trip_distance", "fare_amount")).first()
    df.approxQuantile(["fare_amount", "trip_distance"], [0.25, 0.5, 0.75], 0.01)
    (df.groupBy(F.hour("tpep_pickup_datetime").alias("h"))
       .agg(F.count("*"), F.avg("fare_amount")).collect())


def pandas_clean(path):
    df = pd.read_parquet(path)
    df = df.drop_duplicates()
    df = df[(df.fare_amount > 0) & (df.fare_amount < 500)
            & (df.trip_distance > 0) & (df.trip_distance < 200)
            & (df.tpep_dropoff_datetime > df.tpep_pickup_datetime)].copy()
    for c in FILL_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    df.to_parquet(f"{BENCH_DIR}/_pandas_tmp.parquet")


def timed(fn, *args):
    t = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(BENCH_DIR, exist_ok=True)
    spark = get_spark("Benchmark")
    rows_out = []

    for label, factor in SCALES.items():
        print(f"\n=== Size {label} ===")
        path = make_dataset(spark, label, factor)
        n_rows = spark.read.parquet(path).count()
        mb = folder_mb(path)
        print(f"rows={n_rows}, size={mb} MB")

        stage_profile(spark, path)  # warm-up run, not recorded

        stages = [("profile", stage_profile), ("clean", stage_clean),
                  ("eda", stage_eda)]
        for name, fn in stages:
            for run in range(1, REPEATS + 1):
                sec = timed(fn, spark, path)
                print(f"spark {name} run {run}: {sec:.1f}s")
                rows_out.append({"size": label, "rows": n_rows, "size_mb": mb,
                                 "engine": "spark", "stage": name,
                                 "run": run, "seconds": round(sec, 2)})

        if factor <= PANDAS_MAX_SCALE:
            for run in range(1, REPEATS + 1):
                try:
                    sec = timed(pandas_clean, path)
                    print(f"pandas clean run {run}: {sec:.1f}s")
                    rows_out.append({"size": label, "rows": n_rows, "size_mb": mb,
                                     "engine": "pandas", "stage": "clean",
                                     "run": run, "seconds": round(sec, 2)})
                except MemoryError:
                    print("pandas ran out of memory")
                    break

    spark.stop()

    res = pd.DataFrame(rows_out)
    res.to_csv(RESULTS_CSV, index=False)
    summary = (res.groupby(["engine", "stage", "size", "rows", "size_mb"])["seconds"]
                  .agg(["mean", "std"]).round(2).reset_index()
                  .sort_values(["engine", "stage", "rows"]))
    summary.to_csv(SUMMARY_CSV, index=False)
    print("\n", summary.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        for (eng, stage), g in summary.groupby(["engine", "stage"]):
            plt.plot(g["rows"] / 1e6, g["mean"], marker="o", label=f"{eng} {stage}")
        plt.xlabel("Rows (millions)")
        plt.ylabel("Seconds (mean of runs)")
        plt.title("Processing time vs data size")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(CHART_PNG, dpi=150)
        print("Saved", CHART_PNG)
    except Exception as e:
        print("Chart skipped:", e)

    print("Machine:", platform.processor(), "| CPUs:", os.cpu_count(),
          "| Python", platform.python_version(), "| Spark driver memory 4g")


if __name__ == "__main__":
    main()