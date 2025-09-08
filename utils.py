import numpy as np
import numpy.typing as npt

def compute_centroidDistances(
    x: npt.NDArray, 
    centroids: npt.NDArray
) -> npt.NDArray:
    return np.sum((centroids - x)**2, axis = 1)

def get_minDistance(
    centroidDistances: npt.NDArray
) -> npt.ArrayLike:
    return np.min(centroidDistances)

def get_clusterId(
    centroidDistances: npt.NDArray
) -> npt.ArrayLike:
    return np.argmin(centroidDistances)