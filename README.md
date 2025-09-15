# Distributed-K-means

We implement the k-Means|| and Mini-batch k-Means|| algorithms in Spark's distributed framework.

We evaluate the algorithm’s performance using a Gaussian Mixture standard synthetic dataset, and the kddcup99 dataset (https://scikit-learn.org/stable/datasets/real_world.html#kddcup99-dataset) available on Scikit-learn.

## 1. Clustering - k-Means||

The k-Means method is widely utilized for unsupervised clustering tasks. One common technique for weight initialization in k-Means is known as k-Means++, although it’s primarily a sequential algorithm.

A scalable version, referred to as k-Means|| (parallel k-Means), has been introduced and described in detail in the paper available at: https://arxiv.org/abs/1203.6402.

At the heart of k-Means|| lies the initialization procedure, which is depicted in the following pseudo-code as presented in the aforementioned paper:

Algorithm 2: k-means|| (k, ℓ) initialization.

1:   $C \leftarrow$ sample a point uniformly at random from $X$
2:   $\psi \leftarrow \phi_X(C)$
3:   for $O(\log{\psi}) times do  
4:     $C′ \leftarrow$ sample each point $x\in X$ independently with probability $p_x = \ell \cdot d^2(x,C)/\phi_X(C)$ 
5:     $C \leftarrow C \\cup C'$
6:   end for  
7:   For $x\in C$, set $w_x$ to be the number of points in $X$ closer to $x$ than any other point in $C$
8:   Recluster the weighted points in $C$ into $k$ clusters

## 2. Clustering - Mini-batch k-Means||

Another alternative to k-Means++ and k-Means|| is the Mini-batch k-Means algorithm, detailed in this paper:
https://sci-hub.se/10.1145/1772690.1772862

This approach employs small (mini) batches to optimize k-Means clustering instead of relying on a single large-batch optimization procedure. It has been demonstrated to offer faster convergence and can be implemented to scale k-Means with low computation cost on large datasets.

------------
We implement and benchmark the above mentioned algorithms using Spark's distributed framework.
