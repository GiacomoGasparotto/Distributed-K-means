import os

from pyspark import SparkConf
from pyspark.sql import SparkSession

def sparkSetup(
    appName: str       
) -> SparkSession:
    """
    Returns a `SparkSession` properly configured.
    """
    env_path = "environment.tar.gz"
    env_path = os.path.abspath(env_path)
    conf = SparkConf() \
        .set("spark.archives", f"{env_path}#environment") \
        .set("spark.executor.memory", "4096m")

    # when in docker .master("spark://spark-master:7077")
    spark = SparkSession \
        .builder \
        .master("spark://master:7077") \
        .appName(appName) \
        .config(conf = conf) \
        .getOrCreate()
    return spark