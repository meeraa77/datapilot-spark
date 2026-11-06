from pyspark.sql import functions as F


def read_raw(spark, path):
    if path.endswith(".csv"):
        return spark.read.csv(path, header=True, inferSchema=True)
    return spark.read.parquet(path)


def prepare_taxi(df):
    return df


def prepare_flights(df):
    """Target: 1 = arrived more than 15 minutes late, else 0."""
    return df.withColumn(
        "delayed",
        F.when(F.col("ARRIVAL_DELAY").isNull(), F.lit(None).cast("int"))
         .when(F.col("ARRIVAL_DELAY") > 15, F.lit(1))
         .otherwise(F.lit(0)))


DATASETS = {
    "taxi": {
        "raw": "data/raw/yellow_tripdata_2026-01.parquet",
        "prepare": prepare_taxi,
        "target": "fare_amount",
        "code_cols": ["VendorID", "RatecodeID", "payment_type",
                      "PULocationID", "DOLocationID"],
        "fill_cols": ["passenger_count", "RatecodeID",
                      "congestion_surcharge", "Airport_fee"],
        "zero_fill_cols": [],
        "drop_cols": [],
        "leakage": ["total_amount", "tip_amount", "tolls_amount", "mta_tax",
                    "extra", "improvement_surcharge", "congestion_surcharge",
                    "Airport_fee", "cbd_congestion_fee"],
        "hist_cols": ["fare_amount", "trip_distance"],
        "scatter": ["trip_distance", "fare_amount"],
        "hour": lambda: F.hour("tpep_pickup_datetime"),
    },
    "flights": {
        "raw": "data/raw/flights.parquet",
        "prepare": prepare_flights,
        "target": "delayed",
        "code_cols": ["FLIGHT_NUMBER", "DAY", "MONTH", "DAY_OF_WEEK"],
        "fill_cols": ["DEPARTURE_TIME", "TAXI_OUT", "WHEELS_OFF",
                      "SCHEDULED_TIME", "ELAPSED_TIME", "AIR_TIME",
                      "WHEELS_ON", "TAXI_IN", "ARRIVAL_TIME"],
        "zero_fill_cols": ["AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
                           "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"],
        "drop_cols": ["CANCELLATION_REASON"],
        "leakage": ["ARRIVAL_DELAY", "ARRIVAL_TIME", "WHEELS_ON", "TAXI_IN",
                    "ELAPSED_TIME", "AIR_TIME", "AIR_SYSTEM_DELAY",
                    "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY",
                    "WEATHER_DELAY", "CANCELLED", "DIVERTED",
                    "CANCELLATION_REASON"],
        "hist_cols": ["ARRIVAL_DELAY", "DISTANCE"],
        "scatter": ["DISTANCE", "ARRIVAL_DELAY"],
        "hour": lambda: F.floor(F.col("SCHEDULED_DEPARTURE") / 100).cast("int"),
    },
}