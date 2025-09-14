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
) -> float:
    raise TypeError("Unsupported data type")

@compute_cost.register(RDD)
def _(
    data: RDD, 
    centroids: npt.NDArray
) -> float:
    minDistance_rdd = data \
        .map(lambda x: (x, get_minDistance(compute_centroidDistances(x, centroids))))
    cost = minDistance_rdd \
        .map(lambda x: x[1]) \
        .sum()
    cost /= data.count()
    return float(cost)

@compute_cost.register(np.ndarray)
def _(
    data: npt.NDArray,
    centroids: npt.NDArray
) -> float:
    minDistance = get_minDistance(compute_centroidDistances(data, centroids))
    cost = np.sum(minDistance) / data.shape[0]
    return cost

def early_stop(
        data: RDD | npt.NDArray,
        e: int,
        old_centroids: npt.NDArray,
        centroids: npt.NDArray,
        stop_centroids: bool = False,   
        stop_cost: float = 1e-4,
        check_cost_every: int = 1, 
) -> bool:
    """
    Methods to assess convergence:
    1) Centroid movement -> stop when maximum centroid displacement falls below a tolerance
    2) Cost's relative improvement across iterations -> stop when relative decrease in psi_X(C) below a tolerance
    3) Stop when no labels change
    """
    # 1) centroid-movement convergence
    if stop_centroids: 
        if np.allclose(centroids, old_centroids): return True
    # 2) cost-improvement convergence
    if stop_cost > 0.0 and ((e + 1) % check_cost_every == 0):
        prev_cost = compute_cost(data, old_centroids)
        cur_cost  = compute_cost(data, centroids)
        if prev_cost is not None:
            rel_impr = (prev_cost - cur_cost) / max(prev_cost, 1e-12)
            if rel_impr <= stop_cost:
                return True
    # 3) TBC...
    return False