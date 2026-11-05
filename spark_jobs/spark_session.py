import os
import sys
from pyspark.sql import SparkSession

# Windows needs winutils for writing files
if sys.platform.startswith("win") and os.path.exists("C:/hadoop/bin/winutils.exe"):
    os.environ.setdefault("HADOOP_HOME", "C:/hadoop")
    os.environ["PATH"] = "C:/hadoop/bin;" + os.environ["PATH"]


def get_spark(app_name="DataPilot"):
    return (SparkSession.builder
            .master("local[*]")
            .appName(app_name)
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())