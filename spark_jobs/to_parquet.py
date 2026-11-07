import sys
from spark_jobs.spark_session import get_spark


def run(csv_path="data/raw/flights/flights.csv",
        out_path="data/raw/flights.parquet"):
    spark = get_spark("ToParquet")
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    df.write.mode("overwrite").parquet(out_path)
    saved = spark.read.parquet(out_path)
    print("ROWS:", saved.count())
    print("COLUMNS:", saved.columns)
    spark.stop()


if __name__ == "__main__":
    run(*sys.argv[1:])