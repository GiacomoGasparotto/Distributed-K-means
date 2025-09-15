import numpy as np
import numpy.typing as npt
import os

# dataset
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler

from pyspark import SparkConf
from pyspark.sql import SparkSession

def sparkSetup(
    appName: str,
    env_path: str = "environment.tar.gz",
    executor_memory: int = 4096
) -> SparkSession:
    """
    Returns a `SparkSession` properly configured.
    """
    env_path = os.path.abspath(env_path)
    conf = SparkConf() \
        .set("spark.archives", f"{env_path}#environment") \
        .set("spark.executor.memory", f"{executor_memory}m")

    # when in docker .master("spark://spark-master:7077")
    spark = SparkSession \
        .builder \
        .master("spark://master:7077") \
        .appName(appName) \
        .config(conf = conf) \
        .getOrCreate()
    return spark