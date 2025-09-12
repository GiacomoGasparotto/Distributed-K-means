import numpy as np
import numpy.typing as npt

from functools import singledispatch

from pyspark.rdd import RDD

# --- BASIC OPERATIONS ---

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

def cost_function(
    data: RDD, 
    centroids: npt.ArrayLike
) -> float:
    minDistance_rdd = data \
        .map(lambda x: (x, get_minDistance(compute_centroidDistances(x, centroids))))
    cost = minDistance_rdd \
        .map(lambda x: x[1]) \
        .sum()
    return cost

# --- INITIALIZE CENTROIDS ---

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
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape[0] != data.shape[0]:
            raise ValueError("weights length must match number of data points")
    
    centroids = data[np.random.randint(0, data.shape[0]),:].reshape(1, -1) # reshaping for easier stacking
    
    while (centroids.shape[0] < k):
        # since the original functions are made for map
        # we need to loop over the data
        minDistance_array = np.array(
            [get_minDistance(compute_centroidDistances(datum, centroids)) for datum in data],
            dtype=float
        )
        # Multiply by the weight simulates multiple copies of the same datum
        minDistance_array = minDistance_array * weights
        
        total_minDistance = np.sum(minDistance_array)
        # sampling probability proportional to minDistance
        if not np.isfinite(total_minDistance) or np.isclose(total_minDistance, 0):
            # Fallback to uniform probabilities to avoid division by zero
            probs = np.ones_like(minDistance_array) / minDistance_array.shape[0]
        else:
            probs = (minDistance_array / total_minDistance).ravel()
        new_centroid_idx = np.random.choice(minDistance_array.shape[0], size=1, p=probs)
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
    r: int = 0,
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

    if r == 0: 
        iterations = int(np.ceil(np.log(cost))) if (cost > 1) else 1
    else: 
        iterations = r

    for _ in range(iterations):
        new_centroids = np.array(
            minDistance_rdd \
                .filter(lambda x: np.random.rand() < np.min((l * x[1] / cost, 1))) \
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
    
    minDistance_rdd.unpersist()
    clusterCounts = data_rdd \
        .map(lambda x: (get_clusterId(compute_centroidDistances(x, centroids)), 1)) \
        .countByKey()
    
    clusterCounts = np.array([w[1] for w in clusterCounts.items()])
    centroids = naiveKMeans(
        centroids, 
        kMeansPlusPlus_init(centroids, k, clusterCounts)
    )
    
    return centroids

# --- UPDATE CENTROIDS ---
@singledispatch
def early_stop(
        data: RDD,
        e: int,
        old_centroids: npt.ArrayLike,
        centroids: npt.ArrayLike,
        stop_centroids: float = 0,   
        stop_cost: float = 1e-4,
        check_cost_every: int = 1, 
) -> bool:
    raise TypeError("Unsupported data type")

@early_stop.register
def _(
        data: RDD,
        e: int,
        old_centroids: npt.ArrayLike,
        centroids: npt.ArrayLike,
        stop_centroids: float = 0,   
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
    if stop_centroids > 0.0:
        max_shift = np.linalg.norm(centroids - old_centroids, axis=1).max()
        if max_shift <= stop_centroids:
            return centroids

    # 2) cost-improvement convergence
    if stop_cost > 0.0 and ((e + 1) % check_cost_every == 0):
        prev_cost = cost_function(data, old_centroids)
        cur_cost  = cost_function(data, centroids)
        if prev_cost is not None:
            rel_impr = (prev_cost - cur_cost) / max(prev_cost, 1e-12)
            if rel_impr <= stop_cost:
                return True
    
    # 3) TBC...

@singledispatch
def naiveKMeans(
    data: RDD | npt.NDArray,
    centroids: npt.NDArray,
    epochs: int = 10
) -> npt.NDArray:
    raise TypeError("Unsupported data type")

@naiveKMeans.register(np.ndarray)
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
        assignments = np.array([get_clusterId(compute_centroidDistances(x, centroids)) 
                                for x in data])
        newC = np.empty_like(centroids)
        for i in range(k):
            mask = (assignments == i)
            if np.any(mask):
                newC[i] = data[mask, :].mean(axis=0)
            else:
                
                newC[i] = centroids[i]
        if np.allclose(newC, centroids):
            break
        centroids = newC
    return centroids

@naiveKMeans.register(RDD)
def _(
    data_rdd: RDD,
    centroids: npt.NDArray,
    epochs: int = 10,
    stop_centroids: float = 1e-4,   
    stop_cost: float = 1e-4,
    check_cost_every: int = 1, 
) -> npt.NDArray:
    """
    Standard kMeans algorithm parallel implementation:
    given `data`, updates the (k) `centroids` for `epochs` times,
    improving the clustering each time, check of max centroid shift tolerance to assess convergence
    """
    k = centroids.shape[0]
    prev_cost = None
    
    for e in range(epochs):
        # assign each point to the closest cluster
        data_assigned_rdd = data_rdd \
            .map(lambda x: (get_clusterId(compute_centroidDistances(x, centroids)), 1, x)) \
            .persist()
        
        # counting how many assigments per cluster
        clusterWeights_dict = data_assigned_rdd \
            .map(lambda x: (x[0], x[1])) \
            .countByKey()
        
        # compute the numerator term
        clusterSums_dict = dict(data_assigned_rdd \
                .map(lambda x: (x[0], x[2])) \
                .reduceByKey(lambda x, y: x + y) \
                .collect()
        )

        # compute the weighted average (they are the updated clusters). 
        # If no counts maintain the older centroid values
        old_centroids = centroids
        centroids = np.array(
                [clusterSums_dict[i]/clusterWeights_dict[i] 
                if i in clusterWeights_dict.keys() else centroids[i,:]
                for i in range(k)]
            )
        
        # free memory
        data_assigned_rdd.unpersist()

        # check convergence
        if early_stop(data_rdd, e, old_centroids, centroids, stop_centroids, stop_cost, check_cost_every): 
            return centroids

    return centroids


def miniBatchKMeans(
    data_rdd: RDD,
    centroids: npt.NDArray,
    iterations: int = 10,
    batch_fraction: float = 0.1,
    stop_centroids: float = 1e-4,   
    stop_cost: float = 1e-4,
    check_cost_every: int = 1,
) -> npt.NDArray:
    """
    Mini-batch K-Means implementation with exponential averaging for centroid updates.
    """
    k = centroids.shape[0]
    clusterCounters = np.zeros((k,)) # 1 / learning_rate
    for iter in range(iterations):
        miniBatch_rdd = data_rdd \
            .sample(withReplacement=False, fraction=batch_fraction)
        miniBatch_rdd = miniBatch_rdd \
            .map(lambda x: (get_clusterId(compute_centroidDistances(x, centroids)), 1, x)) \
            .persist()
        
        # counting how many assigments per cluster
        clusterCounts_dict = miniBatch_rdd \
            .map(lambda x: (x[0], x[1])) \
            .countByKey()
        clusterCounts = np.array(
            [clusterCounts_dict[i] if i in clusterCounts_dict.keys() else 0 for i in range(k)]
        )
        clusterCounters += clusterCounts
        
        # edge case in which a cluster has no assignments:
        # if also its counter is zero the whole iteration is repeated
        if any(np.isclose(v, 0) for v in clusterCounters): 
            iter -= 1
            miniBatch_rdd.unpersist()
            continue
        # otherwise its count will be set to 1 to avoid division by 0 in the update step
        clusterCounts = np.where(clusterCounts >= 1, clusterCounts, 1)

        # summing all points assigned to the same cluster
        # (in the update step this will be divided by the counts 
        # in order to get the mean for every cluster).
        # A dict is used for convenience and consistency with clusterCounts
        clusterSums_dict = dict(miniBatch_rdd \
            .map(lambda x: (x[0], x[2])) \
            .reduceByKey(lambda x, y: x + y) \
            .collect()
        )
        # edge case in which a cluster has no assignments:
        # the centroid is returned instead of 0 
        # (which would have been the sum of its assigned points) 
        # in order to not update its position 
        # (note how the terms cancel out in the update step)
        clusterSums = np.array(
            [clusterSums_dict[i] if i in clusterSums_dict.keys() else centroids[i,:] for i in range(k)]
        )
        # store old centroids for early stop
        old_centroids = centroids
        # update step: c <- (1 - eta) * c + eta * x_mean
        # (note x_mean = x_sums / c_count)
        centroids = (1 - 1 / clusterCounters).reshape(-1, 1) * centroids + \
                    (1 / (clusterCounters * clusterCounts)).reshape(-1, 1) * clusterSums
        
        miniBatch_rdd.unpersist()

        # check convergence
        if early_stop(data_rdd, iter, old_centroids, centroids, stop_centroids, stop_cost, check_cost_every): 
            return centroids
        
    return centroids
