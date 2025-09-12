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
    return cost

@compute_cost.register(np.ndarray)
def _(
    data: npt.NDArray,
    centroids: npt.NDArray
) -> float:
    minDistance = np.array(
        [get_minDistance(compute_centroidDistances(x, centroids)) for x in data]
    )
    cost = np.sum(minDistance) / data.shape[0]
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
    centroids = lloydKMeans(
        centroids, 
        kMeansPlusPlus_init(centroids, k, clusterCounts)
    )
    
    return centroids

# --- UPDATE CENTROIDS ---

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
        assignments = np.array(
            [get_clusterId(compute_centroidDistances(x, centroids)) for x in data]
        )
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

# def miniBatchKMeans(
#     data_rdd: RDD,
#     centroids: npt.NDArray,
#     epochs: int = 10,
#     batch_fraction: float = 0.1
# ) -> npt.NDArray:
#     """
#     Mini-batch K-Means implementation with exponential averaging for centroid updates.
#     """
#     k = centroids.shape[0]
#     clusterCounters = np.zeros((k,)) # 1 / learning_rate
#     for iter in range(epochs):
#         miniBatch_rdd = data_rdd \
#             .sample(withReplacement=False, fraction=batch_fraction)
#         miniBatch_rdd = miniBatch_rdd \
#             .map(lambda x: (get_clusterId(compute_centroidDistances(x, centroids)), 1, x)) \
#             .persist()
#         
#         # counting how many assigments per cluster
#         clusterCounts_dict = miniBatch_rdd \
#             .map(lambda x: (x[0], x[1])) \
#             .countByKey()
#         clusterCounts = np.array(
#             [clusterCounts_dict[i] if i in clusterCounts_dict.keys() else 0 for i in range(k)]
#         )
#         clusterCounters += clusterCounts
#         
#         # edge case in which a cluster has no assignments:
#         # if also its counter is zero the whole iteration is repeated
#         if any(np.isclose(v, 0) for v in clusterCounters): 
#             iter -= 1
#             miniBatch_rdd.unpersist()
#             continue
#         # otherwise its count will be set to 1 to avoid division by 0 in the update step
#         clusterCounts = np.where(clusterCounts >= 1, clusterCounts, 1)

#         # summing all points assigned to the same cluster
#         # (in the update step this will be divided by the counts 
#         # in order to get the mean for every cluster).
#         # A dict is used for convenience and consistency with clusterCounts
#         clusterSums_dict = dict(miniBatch_rdd \
#             .map(lambda x: (x[0], x[2])) \
#             .reduceByKey(lambda x, y: x + y) \
#             .collect()
#         )
#         # edge case in which a cluster has no assignments:
#         # the centroid is returned instead of 0 
#         # (which would have been the sum of its assigned points) 
#         # in order to not update its position 
#         # (note how the terms cancel out in the update step)
#         clusterSums = np.array(
#             [clusterSums_dict[i] if i in clusterSums_dict.keys() else centroids[i,:] for i in range(k)]
#         )

#         # update step: c <- (1 - eta) * c + eta * x_mean
#         # (note x_mean = x_sums / c_count)
#         centroids = (1 - 1 / clusterCounters).reshape(-1, 1) * centroids + \
#                     (1 / (clusterCounters * clusterCounts)).reshape(-1, 1) * clusterSums
#         
#         miniBatch_rdd.unpersist()
#         
#     return centroids

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
        # if its also the first iteration: repeat the iteration
        state = False
        for i in range(k):
            if i in clusterMetrics.keys(): continue
            clusterMetrics[i] = (0, centroids[i,:])
            if iter==0:
                state = True
                break
        if state: 
            iter -= 1
            continue

        clusterCounts = np.zeros(shape = (k,))
        clusterSums = np.zeros_like(centroids)
        for i in range(k):
            clusterCounters[i] += clusterMetrics[i][0]
            clusterCounts[i] = np.min((clusterMetrics[i][0],1))
            clusterSums[i,:] = clusterMetrics[i][1]
        
        # update step: c <- (1 - eta) * c + eta * x_mean
        # (note x_mean = x_sums / c_count)
        centroids = (1 - 1 / clusterCounters).reshape(-1, 1) * centroids + \
                    (1 / (clusterCounters * clusterCounts)).reshape(-1, 1) * clusterSums
        
    return centroids