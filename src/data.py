import numpy as np
import numpy.typing as npt
import os

# dataset
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler


def get_kdd(
    standardize: bool = True
) -> tuple[npt.NDArray, npt.NDArray, dict]:
    # fetch the dataset and its labels
    kddData, kddLabels = fetch_kddcup99(
    percent10 = True,
    shuffle = True,
    return_X_y = True
    )
    # transform bytes entries into integers
    entries_dict = {
        i: np.unique(kddData[:,i], return_inverse=True) 
        for i in range(kddData.shape[1]) 
        if isinstance(kddData[0,i], bytes) 
    }
    for key, values in entries_dict.items():
        kddData[:,key] = values[1]
    # and then cast everything into a float
    kddData = kddData.astype(np.float32)

    # standardizing the dataset
    if standardize:
        scaler = StandardScaler()
        kdd_data = scaler.fit_transform(kddData)
    return kddData, kddLabels, entries_dict

def get_gm(
    n: int = 50,  
    k: int = 10,
    dim: int = 15,                  
    R: int = 10,
    standardize: bool = True,
    seed: int = 42
) -> tuple[npt.NDArray, npt.NDArray]:
    np.random.seed(seed)
    # Centers generation N(0, R*I)
    gmCenters = np.random.normal(loc=0, scale=np.sqrt(R), size=(k, dim)).astype(np.float32)
    # Point generation N(center, I) for each cluster
    gmData = np.concatenate(
        [center + np.random.randn(n, dim) for center in gmCenters],
        axis=0
    ).astype(np.float32)
    if standardize:
        scaler = StandardScaler().fit(gmData)
        gmData = scaler.transform(gmData).astype(np.float32)
        gmCenters = scaler.transform(gmCenters).astype(np.float32)
    return gmData, gmCenters