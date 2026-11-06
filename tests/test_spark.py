from spark_jobs.spark_session import get_spark

spark = get_spark()
print("Spark version:", spark.version)
print("Row count:", spark.range(1000).count())
spark.stop()