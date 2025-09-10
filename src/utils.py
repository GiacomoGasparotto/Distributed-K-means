import numpy as np
import numpy.typing as npt
import os

# dataset
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler

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
        .set("spark.executor.memory", "4096m") \
        .set("spark.archives", f"{env_path}#environment")

    # when in docker .master("spark://spark-master:7077")
    spark = SparkSession \
        .builder \
        .master("spark://master:7077") \
        .appName(appName) \
        .config(conf = conf) \
        .getOrCreate()
    return spark

def kddSetup(
    standardize: bool = True
) -> tuple[npt.NDArray, npt.NDArray, dict]:
    # fetch the dataset and its labels
    kdd_data, kdd_labels = fetch_kddcup99(
    percent10 = True,
    shuffle = True,
    return_X_y = True
    )
    # transform bytes entries into integers
    entries_dict = {
    i: np.unique(kdd_data[:,i], return_inverse=True) 
    for i in range(kdd_data.shape[1]) 
        if isinstance(kdd_data[0,i], bytes) 
    }
    for key, values in entries_dict.items():
        kdd_data[:,key] = values[1]
    # and then cast everything into a float
    kdd_data = kdd_data.astype(float)

    # standardizing the dataset
    if standardize:
        scaler = StandardScaler()
        kdd_data = scaler.fit_transform(kdd_data)
    return kdd_data, kdd_labels, entries_dict