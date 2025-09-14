import numpy as np
import numpy.typing as npt

from functools import singledispatch

from pyspark.rdd import RDD

from .base import compute_centroidDistances, get_clusterId, get_minDistance
from .update import lloydKMeans

@singledispatch
def kMeansRandom_init(
    data: RDD | npt.NDArray ,
    k: int
) -> npt.NDArray:
    """
    Initialize `k` centroids taking random points from `data`.
    """
    raise TypeError("Unsupported data type")

@kMeansRandom_init.register(RDD)
def _(
    data: RDD,
    k: int
) -> npt.NDArray:
    centroids = np.array(
        data.takeSample(withReplacement=False, num=k)
    )
    return centroids

@kMeansRandom_init.register(np.ndarray)
def _(
    data: npt.NDArray,
    k: int
) -> npt.NDArray:
    centroids = data[np.random.choice(data.shape[0], size = k, replace = False), :]
    return centroids

def kMeansPlusPlus_init(
    data: npt.NDArray,
    k: int,
    weights: npt.NDArray = np.array([])
) -> npt.NDArray:
    """
    Standard kMeans++ initialization method:
    given `data` (eventually weighted), returns `k` cluster centroids
    """
    # Ensure weights is a 1D array aligned with data points
    if weights.size == 0:
        weights = np.ones(shape=(data.shape[0],), dtype=float)
    else:
        weights = weights.reshape(-1,)
        if weights.shape[0] != data.shape[0]:
            raise ValueError("`weights` length must match number of data points")
    
    centroids = kMeansRandom_init(data, 1).reshape(1, -1) # reshaping for easier stacking
    
    while (centroids.shape[0] < k):
        minDistance = get_minDistance(compute_centroidDistances(data, centroids)) * weights
        total_minDistance = np.sum(minDistance)

        # sampling probability proportional to minDistance
        if (not np.isfinite(total_minDistance)) or np.isclose(total_minDistance, 0):
            # Fallback to uniform probabilities to avoid division by zero
            minDistance = np.ones_like(minDistance)
            total_minDistance = np.sum(minDistance)
        # numpy memory management trick:
        # in this way `probs` is just a view of `minDistance`.
        # If we were using instead
        # `probs = (minDistance / total_minDistance).reshape(-1)`,
        # then `probs` would have been stored as a different array
        minDistance /= total_minDistance # transformation into probabilities
        probs = minDistance.reshape(-1)

        new_centroid_idx = np.random.choice(probs.shape[0], size=1, p=probs)
        new_centroid = data[new_centroid_idx,:].reshape(1, -1)

        # edge case in which the same centroid is selected twice:
        # redo the iteration without saving the centroid
        if any(np.array_equal(new_centroid, row) for row in centroids): continue
        centroids = np.concatenate((centroids, new_centroid), axis = 0)
        
    return centroids

def kMeansParallel_init(
    data_rdd: RDD,
    k: int,
    l: float,
    r: int = 0
) -> npt.NDArray:
    """
    kMeans|| initialization method:
    returns `k` good `centroids`.
    `l` controls the probability of each point
    in `data_rdd` of being sampled as a pre-processed centroid.
    `r` fixes # of iterations if set !=0
    """

    centroids = np.array(
        data_rdd.takeSample(num=1, withReplacement=False)
    )
    
    minDistance_rdd = data_rdd \
        .map(lambda x: (x, get_minDistance(compute_centroidDistances(x, centroids)))) \
        .persist()

    cost = minDistance_rdd \
        .map(lambda x: x[1]) \
        .sum()

    if r < 1: 
        iterations = int(np.ceil(np.log(cost))) if (cost > 1) else 1
    else: 
        iterations = r

    iter = 0
    # edge case in which centroids.shape[0] < k at the end of the iterations:
    # continue until we have enough centroids
    while (iter < iterations) or (centroids.shape[0] < k):
        new_centroids = np.array(
            minDistance_rdd \
                .filter(lambda x: np.random.rand() < np.min([l * x[1] / cost, 1])) \
                .map(lambda x: x[0]) \
                .collect()
        )
        # edge case in which no new centroid is sampled:
        # this avoids the following `np.concatenate` to fail
        if len(new_centroids.shape) < 2:
            continue

        minDistance_rdd.unpersist()
        centroids = np.unique(
            np.concatenate((centroids, new_centroids), axis = 0), 
            axis = 0
        )
        
        minDistance_rdd = data_rdd \
            .map(lambda x: (x, get_minDistance(compute_centroidDistances(x, centroids)))) \
            .persist()
        cost = minDistance_rdd \
            .map(lambda x: x[1]) \
            .sum()
        
        iter += 1
    
    minDistance_rdd.unpersist()
    clusterCounts = data_rdd \
        .map(lambda x: (get_clusterId(compute_centroidDistances(x, centroids)), 1)) \
        .countByKey()
    
    clusterCounts = np.array([w[1] for w in clusterCounts.items()])
    centroids = lloydKMeans(
        centroids, 
        kMeansPlusPlus_init(centroids, k, clusterCounts)
    )
    
    return centroids