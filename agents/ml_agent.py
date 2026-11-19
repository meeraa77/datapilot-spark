import json
import os
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   ClusteringEvaluator,
                                   MulticlassClassificationEvaluator,
                                   RegressionEvaluator)
from pyspark.ml.feature import (OneHotEncoder, StandardScaler, StringIndexer,
                                VectorAssembler)
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.sql import functions as F

from spark_jobs.spark_session import get_spark

SEED = 42
MAX_TRAIN_ROWS = 1_000_000   # laptop limit: bigger data is sampled, and we say so
MAX_OHE_CATEGORIES = 50      # columns with more categories are skipped (warning)
MIN_ROWS = 100


def derive_taxi(df):
    df = (df.withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
            .withColumn("pickup_dow", F.dayofweek("tpep_pickup_datetime")))
    return df, ["pickup_hour", "pickup_dow"]


def derive_none(df):
    return df, []


# Per-dataset ML settings. Leakage columns come from profile.json ("leakage_columns").
ML_CONFIG = {
    "taxi": {
        "derive": derive_taxi,
        "extra_exclude": ["tpep_dropoff_datetime"],  # known only after the trip
    },
    "flights": {
        "derive": derive_none,
        # Predict BEFORE departure, so skip anything known after takeoff.
        "extra_exclude": ["DEPARTURE_DELAY", "DEPARTURE_TIME", "TAXI_OUT",
                          "WHEELS_OFF", "TAIL_NUMBER"],
    },
}


def choose_problem(profile):
    """categorical target -> classification, numeric -> regression, none -> clustering."""
    target = profile.get("suggested_target")
    if not target:
        return "clustering"
    kind = next((c["kind"] for c in profile["columns"] if c["name"] == target), None)
    if kind == "numeric":
        return "regression"
    if kind == "categorical":
        return "classification"
    return "clustering"


def select_features(profile, extra_exclude=()):
    """Returns (numeric, categorical, excluded). Never uses target, IDs or leakage."""
    target = profile.get("suggested_target")
    leakage = set(profile.get("leakage_columns", []))
    skip = leakage | set(extra_exclude) | {target}
    numeric, categorical, excluded = [], [], {}
    for c in profile["columns"]:
        name, kind = c["name"], c["kind"]
        if name in skip:
            excluded[name] = "target, leakage, or known only after the event"
        elif kind == "id":
            excluded[name] = "id column"
        elif kind == "datetime":
            excluded[name] = "datetime (use derived features)"
        elif c["distinct"] <= 1:
            excluded[name] = "constant column"
        elif kind == "numeric":
            numeric.append(name)
        elif kind == "categorical":
            if c["distinct"] <= MAX_OHE_CATEGORIES:
                categorical.append(name)
            else:
                excluded[name] = f"too many categories ({c['distinct']})"
    return numeric, categorical, excluded


def build_prep_stages(num_cols, cat_cols):
    stages, inputs = [], list(num_cols)
    for c in cat_cols:
        stages.append(StringIndexer(inputCol=c, outputCol=f"{c}_idx",
                                    handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_oh",
                                    handleInvalid="keep"))
        inputs.append(f"{c}_oh")
    stages.append(VectorAssembler(inputCols=inputs, outputCol="features",
                                  handleInvalid="keep"))
    stages.append(StandardScaler(inputCol="features", outputCol="scaled",
                                 withMean=False, withStd=True))
    return stages


def feature_names(df):
    attrs = df.schema["features"].metadata["ml_attr"]["attrs"]
    items = [a for group in attrs.values() for a in group]
    return [a["name"] for a in sorted(items, key=lambda a: a["idx"])]


def top_importances(model, names, n=10):
    imp = model.featureImportances.toArray()
    pairs = sorted(zip(names, imp), key=lambda p: p[1], reverse=True)[:n]
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]


def eval_regression(pred):
    out = {}
    for key, metric in [("rmse", "rmse"), ("r2", "r2"), ("mae", "mae")]:
        ev = RegressionEvaluator(labelCol="label", predictionCol="prediction",
                                 metricName=metric)
        out[key] = round(ev.evaluate(pred), 4)
    return out


def eval_classification(pred, n_classes):
    out = {}
    for key, metric in [("accuracy", "accuracy"), ("f1", "f1"),
                        ("weighted_precision", "weightedPrecision"),
                        ("weighted_recall", "weightedRecall")]:
        ev = MulticlassClassificationEvaluator(labelCol="label",
                                               predictionCol="prediction",
                                               metricName=metric)
        out[key] = round(ev.evaluate(pred), 4)
    if n_classes == 2:
        ev = BinaryClassificationEvaluator(labelCol="label",
                                           rawPredictionCol="rawPrediction",
                                           metricName="areaUnderROC")
        out["auc"] = round(ev.evaluate(pred), 4)
    return out


def run_clustering(df, num_cols, warnings):
    sample = df.sample(fraction=min(1.0, 200_000 / df.count()), seed=SEED)
    prep = Pipeline(stages=[
        VectorAssembler(inputCols=num_cols, outputCol="features",
                        handleInvalid="skip"),
        StandardScaler(inputCol="features", outputCol="scaled", withMean=True),
    ]).fit(sample)
    data = prep.transform(sample).cache()
    models, best = [], None
    for k in range(2, 7):
        t = time.perf_counter()
        km = KMeans(k=k, seed=SEED, featuresCol="scaled").fit(data)
        score = ClusteringEvaluator(featuresCol="scaled").evaluate(km.transform(data))
        models.append({"name": f"KMeans k={k}",
                       "metrics": {"silhouette": round(score, 4)},
                       "train_seconds": round(time.perf_counter() - t, 1)})
        if best is None or score > best[1]:
            best = (k, score)
    warnings.append("Clustering run on a 200k-row sample.")
    return models, f"KMeans k={best[0]}", []


def run(name="taxi"):
    start = time.perf_counter()
    with open(f"outputs/{name}/profile.json") as f:
        profile = json.load(f)
    cfg = ML_CONFIG[name]
    problem = choose_problem(profile)
    target = profile.get("suggested_target")
    warnings = []

    results = {"dataset_name": name, "problem_type": problem, "target": target,
               "models": [], "best_model": None, "feature_importance": [],
               "warnings": warnings}

    spark = get_spark("ML")
    try:
        df = spark.read.parquet(f"data/processed/{name}")
        total = df.count()
        if total < MIN_ROWS:
            warnings.append(f"Only {total} rows; too few to train a model.")
            raise ValueError("too few rows")

        df, derived = cfg["derive"](df)
        numeric, categorical, excluded = select_features(profile, cfg["extra_exclude"])
        numeric = numeric + derived
        results["features_used"] = {"numeric": numeric, "categorical": categorical}
        results["excluded_columns"] = excluded
        skipped = [c for c, why in excluded.items() if "too many categories" in why]
        if skipped:
            warnings.append("High-cardinality columns skipped: " + ", ".join(skipped))
        print("Problem:", problem, "| target:", target)
        print("Numeric features:", numeric)
        print("Categorical features:", categorical)

        if problem == "clustering":
            models, best, imp = run_clustering(df, numeric, warnings)
            results.update(models=models, best_model=best, feature_importance=imp)
        else:
            df = (df.withColumn("label", F.col(target).cast("double"))
                    .dropna(subset=["label"]))
            n_classes = 0
            if problem == "classification":
                n_classes = df.select("label").distinct().count()
                if n_classes < 2:
                    warnings.append("Target has a single class; cannot train.")
                    raise ValueError("single class")

            if total > MAX_TRAIN_ROWS:
                df = df.sample(fraction=MAX_TRAIN_ROWS / total, seed=SEED)
                warnings.append(
                    f"Sampled about {MAX_TRAIN_ROWS} of {total} rows for training.")

            train, test = df.randomSplit([0.8, 0.2], seed=SEED)
            prep = Pipeline(stages=build_prep_stages(numeric, categorical)).fit(train)
            train_p = prep.transform(train).select("label", "features", "scaled").cache()
            test_p = prep.transform(test).select("label", "features", "scaled").cache()
            names = feature_names(train_p)
            results["rows_train"], results["rows_test"] = train_p.count(), test_p.count()

            if problem == "classification":
                shares = {r["label"]: r["count"]
                          for r in test_p.groupBy("label").count().collect()}
                majority = max(shares.values()) / sum(shares.values())
                results["majority_class_accuracy"] = round(majority, 4)
                if majority > 0.7:
                    warnings.append(
                        "Imbalanced target: judge by F1/AUC, not accuracy alone.")
                candidates = [
                    ("LogisticRegression (baseline)", LogisticRegression(
                        featuresCol="scaled", labelCol="label", maxIter=30), False),
                    ("RandomForestClassifier", RandomForestClassifier(
                        featuresCol="features", labelCol="label",
                        numTrees=30, maxDepth=8, seed=SEED), True),
                ]
            else:
                candidates = [
                    ("LinearRegression (baseline)", LinearRegression(
                        featuresCol="scaled", labelCol="label", maxIter=50), False),
                    ("RandomForestRegressor", RandomForestRegressor(
                        featuresCol="features", labelCol="label",
                        numTrees=30, maxDepth=8, seed=SEED), True),
                ]

            best = None
            for model_name, est, is_tree in candidates:
                t = time.perf_counter()
                model = est.fit(train_p)
                secs = round(time.perf_counter() - t, 1)
                pred = model.transform(test_p)   # held-out data only
                if problem == "regression":
                    metrics = eval_regression(pred)
                    score = -metrics["rmse"]
                else:
                    metrics = eval_classification(pred, n_classes)
                    score = metrics["f1"]
                print(model_name, metrics, f"{secs}s")
                results["models"].append({"name": model_name, "metrics": metrics,
                                          "train_seconds": secs})
                if best is None or score > best[0]:
                    best = (score, model_name)
                if is_tree:
                    results["feature_importance"] = top_importances(model, names)
            results["best_model"] = best[1]

    except Exception as e:  # readable warning, never a crash
        warnings.append(f"ML stage stopped: {e}")
        print("WARNING:", e)

    results["seconds"] = round(time.perf_counter() - start, 1)
    out = f"outputs/{name}/model_results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved", out, "in", results["seconds"], "seconds")
    spark.stop()
    return results


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "taxi")