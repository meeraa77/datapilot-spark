from agents.data_agent import classify


def test_timestamp_is_datetime():
    assert classify("t", "timestamp_ntz", 1000, 1000) == "datetime"


def test_unique_column_is_id():
    assert classify("trip_id", "bigint", 990, 1000) == "id"


def test_many_values_numeric():
    assert classify("fare", "double", 500, 1000) == "numeric"


def test_few_values_categorical():
    assert classify("vendor", "int", 4, 1000) == "categorical"


def test_string_is_categorical():
    assert classify("flag", "string", 3, 1000) == "categorical"