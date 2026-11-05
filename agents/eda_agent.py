import json
import math
import os
import time

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import functions as F

from spark_jobs.spark_session import get_spark

CLEAN_PATH = "data/processed/taxi"
PROFILE_PATH = "outputs/profile.json"
OUT_PATH = "outputs/eda_results.json"
TARGET = "fare_amount"
HIST_COLS = ["fare_amount", "trip_distance"]
BINS = 20

# These are numbers in the file, but they are really category codes
CODE_COLS = {"VendorID", "RatecodeID", "payment_type",
             "PULocationID", "DOLocationID", "store_and_fwd_flag"}


def num(x):
    """Make a value safe for JSON (no NaN / infinity)."""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, 4)


def split_columns(profile):
    numeric, categorical = [], []
    for c in profile["columns"]:
        name, kind = c["name"], c["kind"]
        if name in CODE_COLS:
            categorical.append(name)
        elif kind == "numeric":
            numeric.append(name)
        elif kind == "categorical":
            categorical.append(name)
    return numeric, categorical


def run(clean_path=CLEAN_PATH, profile_path=PROFILE_PATH, out_path=OUT_PATH):
    start = time.perf_counter()
    spark = get_spark("EDA")
    df = spark.read.parquet(clean_path)

    with open(profile_path) as f:
        profile = json.load(f)
    numeric, categorical = split_columns(profile)
    print("Numeric:", numeric)
    print("Categorical:", categorical)

    results = {
        "dataset_name": profile["dataset_name"],
        "row_count": df.count(),
        "target": TARGET,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
    }

    # 1. Summary statistics (one pass) + quartiles
    aggs = []
    for c in numeric:
        aggs += [F.mean(c).alias(f"{c}|mean"), F.stddev(c).alias(f"{c}|std"),
                 F.min(c).alias(f"{c}|min"), F.max(c).alias(f"{c}|max")]
    row = df.agg(*aggs).first().asDict()
    quant = df.approxQuantile(numeric, [0.25, 0.5, 0.75, 0.99], 0.01)

    summary = {}
    for i, c in enumerate(numeric):
        q1, med, q3, p99 = quant[i]
        summary[c] = {
            "mean": num(row[f"{c}|mean"]), "std": num(row[f"{c}|std"]),
            "min": num(row[f"{c}|min"]), "q1": num(q1), "median": num(med),
            "q3": num(q3), "p99": num(p99), "max": num(row[f"{c}|max"]),
        }
    results["summary_stats"] = summary

    # 2. Outliers with the IQR rule (one pass)
    bounds = {}
    out_aggs = []
    for c in numeric:
        q1, q3 = summary[c]["q1"], summary[c]["q3"]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        bounds[c] = (lo, hi)
        out_aggs.append(
            F.sum(((F.col(c) < lo) | (F.col(c) > hi)).cast("int")).alias(c))
    out_row = df.agg(*out_aggs).first().asDict()
    total = results["row_count"]
    results["outliers"] = {
        c: {"lower_bound": num(bounds[c][0]), "upper_bound": num(bounds[c][1]),
            "outlier_count": int(out_row[c] or 0),
            "outlier_percent": round(100 * int(out_row[c] or 0) / total, 2)}
        for c in numeric
    }

    # 3. Top correlations (computed inside Spark)
    assembler = VectorAssembler(inputCols=numeric, outputCol="features",
                                handleInvalid="skip")
    vec = assembler.transform(df).select("features")
    matrix = Correlation.corr(vec, "features", "pearson").head()[0].toArray()
    pairs = []
    for i in range(len(numeric)):
        for j in range(i + 1, len(numeric)):
            v = num(matrix[i][j])
            if v is not None:
                pairs.append({"feature_a": numeric[i], "feature_b": numeric[j],
                              "correlation": v})
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    results["top_correlations"] = pairs[:10]
    results["target_correlations"] = [
        p for p in pairs if TARGET in (p["feature_a"], p["feature_b"])][:10]

    # 4. Top categories per categorical column
    cats = {}
    for c in categorical:
        rows = (df.groupBy(c).count().orderBy(F.desc("count")).limit(10).collect())
        cats[c] = [{"value": r[c], "count": int(r["count"])} for r in rows]
    results["top_categories"] = cats

    # 5. Chart-ready histograms (bins computed in Spark, clipped at the 99th percentile)
    hists = {}
    for c in HIST_COLS:
        if c not in summary:
            continue
        lo, hi = summary[c]["min"], summary[c]["p99"]
        width = (hi - lo) / BINS
        if width <= 0:
            continue
        b = F.least(F.floor((F.col(c) - lo) / width), F.lit(BINS - 1)).cast("int")
        counts = {r["bin"]: int(r["count"])
                  for r in df.select(b.alias("bin")).groupBy("bin").count().collect()}
        hists[c] = [{"bin_start": round(lo + k * width, 3),
                     "bin_end": round(lo + (k + 1) * width, 3),
                     "count": counts.get(k, 0)} for k in range(BINS)]
    results["histograms"] = hists

    # 6. Trend: trips and average fare by pickup hour
    hour_rows = (df.groupBy(F.hour("tpep_pickup_datetime").alias("hour"))
                   .agg(F.count("*").alias("trips"), F.avg(TARGET).alias("avg_fare"))
                   .orderBy("hour").collect())
    results["by_pickup_hour"] = [
        {"hour": r["hour"], "trips": int(r["trips"]), "avg_fare": num(r["avg_fare"])}
        for r in hour_rows]

    # 7. Small random sample for a scatter plot (never the full data)
    sample = (df.select("trip_distance", TARGET).sample(fraction=0.001, seed=42)
                .limit(3000).collect())
    results["scatter_sample"] = [
        {"trip_distance": num(r["trip_distance"]), TARGET: num(r[TARGET])}
        for r in sample]

    results["seconds"] = round(time.perf_counter() - start, 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("Saved", out_path, "in", results["seconds"], "seconds")
    spark.stop()
    return results


if __name__ == "__main__":
    run()