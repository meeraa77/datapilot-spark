from pyspark.sql import SparkSession

def get_spark(app_name="DataPilot"):
    return (SparkSession.builder
            .master("local[*]")
            .appName(app_name)
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())