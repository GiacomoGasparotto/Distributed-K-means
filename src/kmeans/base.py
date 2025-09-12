import numpy as np
import numpy.typing as npt

from functools import singledispatch

from pyspark.rdd import RDD

def compute_centroidDistances(
    x: npt.NDArray, 
    centroids: npt.NDArray
) -> npt.NDArray:
    if len(centroids.shape) != 2:
        raise TypeError("`centroids` has invalid shape")

    if len(x.shape) == 1:
        return np.sum((centroids - x)**2, axis = 1)
    elif len(x.shape) == 2:
        return np.sum((centroids[np.newaxis,:,:] - x[:,np.newaxis,:])**2, axis = 2)
    else:
        raise TypeError("`x` has invalid shape")

def get_minDistance(
    centroidDistances: npt.NDArray
) -> npt.NDArray: 
    return np.min(centroidDistances, axis = -1)

def get_clusterId(
    centroidDistances: npt.NDArray
) -> npt.NDArray:
    return np.argmin(centroidDistances, axis = -1)

@singledispatch
def compute_cost(
    data: RDD | npt.NDArray,
    centroids: npt.NDArray
) -> npt.ArrayLike:
    raise TypeError("Unsupported data type")

@compute_cost.register(RDD)
def _(
    data: RDD, 
    centroids: npt.NDArray
) -> npt.ArrayLike:
    minDistance_rdd = data \
        .map(lambda x: (x, get_minDistance(compute_centroidDistances(x, centroids))))
    cost = minDistance_rdd \
        .map(lambda x: x[1]) \
        .sum()
    cost /= data.count()
    return cost

@compute_cost.register(np.ndarray)
def _(
    data: npt.NDArray,
    centroids: npt.NDArray
) -> npt.ArrayLike:
    minDistance = np.array(
        [get_minDistance(compute_centroidDistances(x, centroids)) for x in data]
    )
    cost = np.sum(minDistance) / data.shape[0]
    return cost