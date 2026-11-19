from agents.ml_agent import choose_problem, select_features

PROFILE = {
    "suggested_target": "fare",
    "leakage_columns": ["total"],
    "columns": [
        {"name": "fare", "kind": "numeric", "distinct": 500},
        {"name": "total", "kind": "numeric", "distinct": 500},
        {"name": "trip_id", "kind": "id", "distinct": 1000},
        {"name": "dist", "kind": "numeric", "distinct": 300},
        {"name": "vendor", "kind": "categorical", "distinct": 3},
        {"name": "zone", "kind": "categorical", "distinct": 260},
        {"name": "when", "kind": "datetime", "distinct": 900},
    ],
}


def test_numeric_target_is_regression():
    assert choose_problem(PROFILE) == "regression"


def test_categorical_target_is_classification():
    p = {"suggested_target": "y", "columns": [
        {"name": "y", "kind": "categorical", "distinct": 2}]}
    assert choose_problem(p) == "classification"


def test_no_target_is_clustering():
    assert choose_problem({"suggested_target": None, "columns": []}) == "clustering"


def test_no_leakage_id_or_target_in_features():
    num, cat, excluded = select_features(PROFILE)
    used = set(num + cat)
    assert "fare" not in used      # the target
    assert "total" not in used     # leakage
    assert "trip_id" not in used   # id
    assert "when" not in used      # datetime
    assert "zone" not in used      # too many categories
    assert num == ["dist"] and cat == ["vendor"]