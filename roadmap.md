# Init analysis

Algorithms:
- kMeansRandom (serial / parallel)
- kMeansPlusPlus
- kMeansParallel (2 values of l at least)

Dataset:
- GM with various variances

Pipeline:
- Test algos with GM
- Plots:
    - Score vs variance
    - time vs algo
    - cost vs r (with kmeans++ as baseline)

# Update analysis

Algorithms:
- lloydKMeans (serial / parallel)
- miniBatchKMeans

Dataset:
- kdd

Pipeline:
- time vs partitions (histogram with lloyd serial baseline and comparison between lloyd parallel and minibatch) 
- cost vs iterations (with lloyd as baseline and with various minibatch fractions)