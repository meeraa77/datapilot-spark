from spark_jobs.spark_session import get_spark

spark = get_spark()
df = spark.read.parquet("data/raw/yellow_tripdata_2026-01.parquet")
print("Rows:", df.count())
print("Columns:", len(df.columns))
df.printSchema()
df.show(5)
spark.stop()