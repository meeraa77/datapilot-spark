from spark_jobs.spark_session import get_spark

spark = get_spark()
df = spark.read.parquet("data/raw/yellow_tripdata_2026-01.parquet")
print("ROWS:", df.count())
print("COLUMNS:", df.columns)
spark.stop()