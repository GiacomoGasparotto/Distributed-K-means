import numpy as np
import numpy.typing as npt

from functools import singledispatch

from pyspark.rdd import RDD

from .base import compute_centroidDistances, get_clusterId, get_minDistance

@singledispatch
def lloydKMeans(
    data: RDD | npt.NDArray,
    centroids: npt.NDArray,
    epochs: int = 10
) -> npt.NDArray:
    raise TypeError("Unsupported data type")

@lloydKMeans.register(np.ndarray)
def _(
    data: npt.NDArray,
    centroids: npt.NDArray,
    epochs: int = 10
) -> npt.NDArray:
    """
    Standard kMeans algorithm serial implementation:
    given `data`, updates the (k) `centroids` for `epochs` times,
    improving the clustering each time
    """
    k = centroids.shape[0]
    for _ in range(epochs):
        assignments = get_clusterId(compute_centroidDistances(data, centroids))
        centroids = np.array(
            [np.mean(data[assignments==i,:], axis = 0) if i in assignments else centroids[i,:] for i in range(k)]
        )
    return centroids

@lloydKMeans.register(RDD)
def _(
    data: RDD,
    centroids: npt.NDArray,
    epochs: int = 10
) -> npt.NDArray:
    """
    Standard kMeans algorithm parallel implementation:
    given `data`, updates the (k) `centroids` for `epochs` times,
    improving the clustering each time
    """
    k = centroids.shape[0]
    for _ in range(epochs):
        clusterMetrics = dict(data \
            .map(lambda x: (get_clusterId(compute_centroidDistances(x, centroids)), (1, x))) \
            .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
            .collect()
        )
        # compute the weighted average (they are the updated clusters). 
        # If no counts maintain the older centroid values
        centroids = np.array(
            [clusterMetrics[i][1]/clusterMetrics[i][0] 
            if i in clusterMetrics.keys() else centroids[i,:]
            for i in range(k)]
        )

    return centroids

def miniBatchKMeans(
    data_rdd: RDD,
    centroids: npt.NDArray,
    epochs: int = 10,
    batch_fraction: float = 0.1
) -> npt.NDArray:
    """
    Mini-batch K-Means implementation with exponential averaging for centroid updates.
    """
    k = centroids.shape[0]
    clusterCounters = np.zeros(shape=(k,)) # 1 / learning_rate
    for iter in range(epochs):
        miniBatch_rdd = data_rdd \
            .sample(withReplacement=False, fraction=batch_fraction)
        clusterMetrics = dict(miniBatch_rdd \
            .map(lambda x: (get_clusterId(compute_centroidDistances(x, centroids)), (1, x))) \
            .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
            .collect()
        )

        # edge case in which a centroid has no assignments
        for i in range(k):
            if i in clusterMetrics.keys(): continue
            clusterMetrics[i] = (0, centroids[i,:])

        clusterCounts = np.zeros(shape = (k,))
        clusterSums = np.zeros_like(centroids)
        for i in range(k):
            clusterCounters[i] += clusterMetrics[i][0]
            clusterCounts[i] = max(clusterMetrics[i][0],1)
            clusterSums[i,:] = clusterMetrics[i][1]
        
        # update step: c <- (1 - eta) * c + eta * x_mean
        # (note x_mean = x_sums / c_count)
        centroids = (1 - 1 / (clusterCounters + 1)).reshape(-1, 1) * centroids + \
                    (1 / ((clusterCounters + 1) * clusterCounts)).reshape(-1, 1) * clusterSums

    return centroids