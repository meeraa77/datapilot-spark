import json
import math
import os

import pytest

from agents.eda_agent import (is_numeric_dtype, num, outliers, summary_stats,
                              top_categories, top_correlations)
from spark_jobs.spark_session import get_spark


@pytest.fixture(scope="module")
def spark():
    return get_spark("test_eda")


@pytest.fixture(scope="module")
def tiny(spark):
    # x = 1..100 plus one extreme value; y = 2 * x; g = a category column.
    # Built with Spark SQL only (no Python workers), so it also runs on
    # Windows machines where Spark cannot find the Python executable.
    base = spark.range(1, 101).selectExpr(
        "cast(id as double) as x",
        "cast(id * 2 as double) as y",
        "case when id % 4 = 0 then 'b' else 'a' end as g")
    extra = spark.sql("select cast(1000.0 as double) as x, cast(2000.0 as double) as y, 'a' as g")
    return base.unionByName(extra)


# ---------------------------------------------------------------- helpers
def test_num_handles_bad_values():
    assert num(None) is None
    assert num(float("nan")) is None
    assert num(float("inf")) is None
    assert num(1.23456789) == 1.2346


def test_is_numeric_dtype():
    assert is_numeric_dtype("double")
    assert is_numeric_dtype("int")
    assert is_numeric_dtype("decimal(10,2)")
    assert not is_numeric_dtype("string")
    assert not is_numeric_dtype("timestamp_ntz")


# ---------------------------------------------------------------- analyses
def test_summary_stats(tiny):
    s = summary_stats(tiny, ["x"])["x"]
    assert s["min"] == 1.0
    assert s["max"] == 1000.0
    assert math.isclose(s["mean"], (5050 + 1000) / 101, abs_tol=1e-3)


def test_outlier_is_found(tiny):
    o = outliers(tiny, ["x"])["x"]
    assert o["method"] == "IQR"
    assert o["count"] == 1


def test_correlation_and_leakage_flag(tiny):
    result = top_correlations(tiny, ["x", "y"], target="y", leakage={"x"})
    assert result[0]["col_a"] == "x"
    assert result[0]["col_b"] == "y"
    assert result[0]["correlation"] == 1.0
    assert result[0]["leakage"] is True


def test_correlation_skips_non_numeric_target(tiny):
    assert top_correlations(tiny, ["x"], target="g", leakage=set()) == []


def test_top_categories(tiny):
    cats = top_categories(tiny, ["g"])["g"]
    assert cats[0] == {"value": "a", "count": 76}
    assert cats[1] == {"value": "b", "count": 25}


# ---------------------------------------------------------------- contract
REQUIRED_KEYS = ["dataset", "row_count", "target", "summary_stats",
                 "top_correlations", "top_categories", "outliers", "chart_data"]


@pytest.mark.parametrize("name", ["taxi", "flights"])
def test_eda_results_follow_contract(name):
    path = f"outputs/{name}/eda_results.json"
    if not os.path.exists(path):
        pytest.skip(f"{path} not generated yet (run python -m agents.eda_agent {name})")
    with open(path) as f:
        e = json.load(f)
    for key in REQUIRED_KEYS:
        assert key in e, f"missing '{key}' in {path}"
    assert e["dataset"] == name
    assert e["row_count"] > 0
    assert len(e["top_correlations"]) > 0
    for h in e["chart_data"]["histograms"].values():
        assert len(h["bins"]) == len(h["counts"]) > 0