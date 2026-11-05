from spark_jobs.clean import remove_invalid, fill_nulls


def make_df(spark, values_sql):
    """values_sql: rows like (10.0, 2.0, '2026-01-01 10:00:00', '2026-01-01 10:30:00')"""
    return spark.sql(f"""
        SELECT col1 AS fare_amount,
               col2 AS trip_distance,
               to_timestamp(col3) AS tpep_pickup_datetime,
               to_timestamp(col4) AS tpep_dropoff_datetime
        FROM VALUES {values_sql}
    """)


T1 = "'2026-01-01 10:00:00'"
T2 = "'2026-01-01 10:30:00'"


def test_valid_row_is_kept(spark):
    df = make_df(spark, f"(10.0, 2.0, {T1}, {T2})")
    assert remove_invalid(df).count() == 1


def test_negative_fare_removed(spark):
    df = make_df(spark, f"(-5.0, 2.0, {T1}, {T2})")
    assert remove_invalid(df).count() == 0


def test_zero_distance_removed(spark):
    df = make_df(spark, f"(10.0, 0.0, {T1}, {T2})")
    assert remove_invalid(df).count() == 0


def test_dropoff_before_pickup_removed(spark):
    df = make_df(spark, f"(10.0, 2.0, {T2}, {T1})")
    assert remove_invalid(df).count() == 0


def test_mixed_rows_counts(spark):
    df = make_df(spark, f"""(10.0, 2.0, {T1}, {T2}),
                            (-1.0, 2.0, {T1}, {T2}),
                            (600.0, 2.0, {T1}, {T2}),
                            (12.0, 3.0, {T1}, {T2})""")
    assert remove_invalid(df).count() == 2


def test_nulls_filled_with_median(spark):
    df = spark.sql("SELECT col1 AS x FROM VALUES (1.0), (3.0), (5.0), (CAST(NULL AS DOUBLE))")
    out, report = fill_nulls(df, ["x"])
    assert out.filter("x IS NULL").count() == 0
    assert report["x"]["nulls_filled"] == 1