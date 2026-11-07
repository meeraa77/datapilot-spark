import json
import os
import platform
import sys
import time

import pandas as pd
from pyspark.sql import functions as F

from spark_jobs.clean import INVALID, fill_nulls
from spark_jobs.datasets import DATASETS, read_raw
from spark_jobs.spark_session import get_spark

# 0.25x = a quarter of the file, 2x = the file stacked twice.
# Add "4x": 4 only if your laptop has 16 GB RAM or more.
SCALES = {"0.25x": 0.25, "1x": 1, "2x": 2}
REPEATS = 5
# Largest scale pandas is attempted at, per dataset (flights has many columns).
PANDAS_MAX_SCALE = {"taxi": 2, "flights": 1}


def make_dataset(spark, cfg, bench_dir, label, factor):
    path = f"{bench_dir}/size_{label}"
    if os.path.exists(path):
        return path
    base = read_raw(spark, cfg["raw"])
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


def stage_clean(spark, path, name, bench_dir):
    cfg = DATASETS[name]
    df = cfg["prepare"](spark.read.parquet(path)).dropDuplicates()
    df = df.dropna(subset=[cfg["target"]])
    df = INVALID[name](df)
    df, _ = fill_nulls(df, cfg["fill_cols"])
    zero = {c: 0 for c in cfg["zero_fill_cols"] if c in df.columns}
    if zero:
        df = df.fillna(zero)
    text = {c: "unknown" for c, t in df.dtypes if t == "string"}
    if text:
        df = df.fillna(text)
    df = df.drop(*[c for c in cfg["drop_cols"] if c in df.columns])
    df.write.mode("overwrite").parquet(f"{bench_dir}/_clean_tmp")


def stage_eda(spark, path, name):
    cfg = DATASETS[name]
    x, y = cfg["scatter"]
    df = spark.read.parquet(path)
    df.agg(F.mean(y), F.stddev(y), F.corr(x, y)).first()
    df.approxQuantile(cfg["hist_cols"], [0.25, 0.5, 0.75], 0.01)
    (df.groupBy(cfg["hour"]().alias("h"))
       .agg(F.count("*"), F.avg(y)).collect())


def pandas_invalid(df, name):
    if name == "taxi":
        return df[(df.fare_amount > 0) & (df.fare_amount < 500)
                  & (df.trip_distance > 0) & (df.trip_distance < 200)
                  & (df.tpep_dropoff_datetime > df.tpep_pickup_datetime)].copy()
    return df[(df.CANCELLED == 0) & (df.DIVERTED == 0)
              & df.ARRIVAL_DELAY.notna() & df.DEPARTURE_DELAY.notna()].copy()


def pandas_clean(path, name, bench_dir):
    cfg = DATASETS[name]
    df = pd.read_parquet(path).drop_duplicates()
    df = pandas_invalid(df, name)
    for c in cfg["fill_cols"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    for c in cfg["zero_fill_cols"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].fillna("unknown")
    df = df.drop(columns=[c for c in cfg["drop_cols"] if c in df.columns])
    df.to_parquet(f"{bench_dir}/_pandas_tmp.parquet")


def timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def main(name):
    cfg = DATASETS[name]
    bench_dir = f"data/benchmark/{name}"
    out_dir = f"outputs/{name}"
    results_csv = f"{out_dir}/benchmark_results.csv"
    summary_csv = f"{out_dir}/benchmark_summary.csv"
    chart_png = f"{out_dir}/scalability.png"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(bench_dir, exist_ok=True)

    spark = get_spark("Benchmark")
    rows_out = []

    for label, factor in SCALES.items():
        print(f"\n=== {name} size {label} ===")
        path = make_dataset(spark, cfg, bench_dir, label, factor)
        n_rows = spark.read.parquet(path).count()
        mb = folder_mb(path)
        print(f"rows={n_rows}, size={mb} MB")

        stages = [
            ("profile", lambda: stage_profile(spark, path)),
            ("clean", lambda: stage_clean(spark, path, name, bench_dir)),
            ("eda", lambda: stage_eda(spark, path, name)),
        ]
        for stage, fn in stages:
            fn()  # warm-up run for this stage and size, not recorded
            for run in range(1, REPEATS + 1):
                sec = timed(fn)
                print(f"spark {stage} run {run}: {sec:.1f}s")
                rows_out.append({"size": label, "rows": n_rows, "size_mb": mb,
                                 "engine": "spark", "stage": stage,
                                 "run": run, "seconds": round(sec, 2)})

        if factor <= PANDAS_MAX_SCALE[name]:
            for run in range(1, REPEATS + 1):
                try:
                    sec = timed(lambda: pandas_clean(path, name, bench_dir))
                    print(f"pandas clean run {run}: {sec:.1f}s")
                    rows_out.append({"size": label, "rows": n_rows, "size_mb": mb,
                                     "engine": "pandas", "stage": "clean",
                                     "run": run, "seconds": round(sec, 2)})
                except MemoryError:
                    print("pandas ran out of memory")
                    break

    spark.stop()

    res = pd.DataFrame(rows_out)
    res.to_csv(results_csv, index=False)
    summary = (res.groupby(["engine", "stage", "size", "rows", "size_mb"])["seconds"]
                  .agg(["mean", "median", "std"]).round(2).reset_index()
                  .sort_values(["engine", "stage", "rows"]))
    summary.to_csv(summary_csv, index=False)
    print("\n", summary.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        for (eng, stage), g in summary.groupby(["engine", "stage"]):
            plt.plot(g["rows"] / 1e6, g["median"], marker="o", label=f"{eng} {stage}")
        plt.yscale("log")
        plt.xlabel("Rows (millions)")
        plt.ylabel("Seconds (median of runs, log scale)")
        plt.title(f"Processing time vs data size ({name})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(chart_png, dpi=150)
        print("Saved", chart_png)
    except Exception as e:
        print("Chart skipped:", e)

    machine = {"processor": platform.processor(), "cpus": os.cpu_count(),
               "python": platform.python_version(), "spark_driver_memory": "4g",
               "repeats": REPEATS, "ram_gb": "FILL IN"}
    with open(f"{out_dir}/benchmark_machine.json", "w") as f:
        json.dump(machine, f, indent=2)
    print("Machine:", machine)


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "taxi"
    if ds not in DATASETS:
        sys.exit(f"Unknown dataset '{ds}'. Choose from: {list(DATASETS)}")
    main(ds)