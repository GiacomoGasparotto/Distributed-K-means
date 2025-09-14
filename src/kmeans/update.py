import numpy as np
import numpy.typing as npt

from functools import singledispatch

from pyspark.rdd import RDD

from .base import compute_centroidDistances, compute_cost, get_clusterId, early_stop

@singledispatch
def lloydKMeans(
    data: RDD | npt.NDArray,
    centroids: npt.NDArray,
    iterations: int = 10,
    save_cost: bool = False,
    earlyStopping: bool = True,
    verbose: bool = False
) -> npt.NDArray | tuple[npt.NDArray, list]:
    raise TypeError("Unsupported data type")

@lloydKMeans.register(np.ndarray)
def _(
    data: npt.NDArray,
    centroids: npt.NDArray,
    iterations: int = 10,
    save_cost: bool = False,
    earlyStopping: bool = True,
    verbose: bool = False
) -> npt.NDArray | tuple[npt.NDArray, list]:
    """
    Standard kMeans algorithm serial implementation:
    given `data`, updates the (k) `centroids` for `iterations` times,
    improving the clustering each time
    """
    costHistory = []
    k = centroids.shape[0]
    for iter in range(iterations):
        assignments = get_clusterId(compute_centroidDistances(data, centroids))
        old_centroids = centroids.copy()
        centroids = np.array(
            [np.mean(data[assignments==i,:], axis = 0) if i in assignments else centroids[i,:] for i in range(k)]
        )
        if save_cost:
            costHistory.append(compute_cost(data, centroids))
        if (earlyStopping and early_stop(data, iter, old_centroids, centroids)):
            if verbose: print(f"CONVERGED! in {iter} iterations") 
            break

    if save_cost: return centroids, costHistory
    return centroids

@lloydKMeans.register(RDD)
def _(
    data: RDD,
    centroids: npt.NDArray,
    iterations: int = 10,
    save_cost: bool = False,
    earlyStopping: bool = True,
    verbose: bool = False
) -> npt.NDArray | tuple[npt.NDArray, list]:
    """
    Standard kMeans algorithm parallel implementation:
    given `data`, updates the (k) `centroids` for `iterations` times,
    improving the clustering each time
    """
    costHistory = []
    k = centroids.shape[0]
    for iter in range(iterations):
        clusterMetrics = dict(data \
            .map(lambda x: (get_clusterId(compute_centroidDistances(x, centroids)), (1, x))) \
            .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
            .collect()
        )
        # store old centroids
        old_centroids = centroids.copy()

        # compute the weighted average (they are the updated clusters). 
        # If no counts maintain the older centroid values
        centroids = np.array(
            [clusterMetrics[i][1]/clusterMetrics[i][0] 
            if i in clusterMetrics.keys() else centroids[i,:]
            for i in range(k)]
        )

        if save_cost:
            costHistory.append(compute_cost(data, centroids))
        if (earlyStopping and early_stop(data, iter, old_centroids, centroids)):
            if verbose: print(f"CONVERGED! in {iter} iterations") 
            break

    if save_cost: return centroids, costHistory
    return centroids

def miniBatchKMeans(
    data_rdd: RDD,
    centroids: npt.NDArray,
    iterations: int = 10,
    batch_fraction: float = 0.1,
    save_cost: bool = False,
    patience: int = 3,
    earlyStopping: bool = True,
    verbose: bool = False
) -> npt.NDArray | tuple[npt.NDArray, list]:
    """
    Mini-batch K-Means implementation with exponential averaging for centroid updates.
    """
    costHistory = []
    k = centroids.shape[0]
    centroidsHistory = []
    clusterCounters = np.zeros(shape=(k,)) # 1 / learning_rate
    for iter in range(iterations):
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
        # store olde centroids
        centroidsHistory.append(centroids)

        if save_cost:
            costHistory.append(compute_cost(data_rdd, centroids))
        if earlyStopping and iter>patience and early_stop(data_rdd, iter, np.mean(centroidsHistory[iter-patience:], axis=0), centroids): 
            if verbose: print(f"CONVERGED! in {iter} iterations") 
            break

    if save_cost: return centroids, costHistory
    return centroids