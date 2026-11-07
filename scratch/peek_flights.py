from spark_jobs.spark_session import get_spark

spark = get_spark()
df = spark.read.csv("data/raw/flights/flights.csv", header=True, inferSchema=True)
print("rows:", df.count())
print("columns:", len(df.columns))
for name, dtype in df.dtypes:
    print(f"{name:25s} {dtype}")