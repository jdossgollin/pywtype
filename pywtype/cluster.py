"""
Implement clustering algorithms
"""

from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans
from numba import jit

class ClusteringAlgorithm:
    """
    A generic class for implementing clustering
    """
    def __init__(self) -> None:
        """
        Set up the clustering algorithm
        """
        self.param = {}
        self.is_fitted = False
        self.score = np.nan
        raise NotImplementedError

    def fit(self) -> None:
        """
        Fit the model
        """
        raise NotImplementedError

    def get_score(self) -> float:
        """
        Get the score of the model
        """
        if not self.is_fitted:
            raise ValueError((
                "Cannot return the score of the model because"
                "the model has not yet been fit"
            ))
        return self.score

def loop_kmeans(data: np.ndarray, n_cluster: int,
                n_init: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the K-means algorithm several times and return all centroids

    Parameters
    ----------
    data : ``np.ndarray``
        the input data
    n_cluster : ``int``
        the number of clusters to run
    n_init : ``int``
        the number of times to run the K-means algorithm

    Returns
    -------
    cluster_centroids : ``np.ndarray``
        array of shape (n_init, n_cluster, n_eofs) with all cluster centroids
    cluster_labels : ``np.ndarray``
        array of shape (n_init, n_obs) with all cluster labels
    """
    n_obs = data.shape[0] # the number of observations
    n_eofs = data.shape[1] # the number of EOFS retained
    cluster_centroids = np.zeros(shape=(n_init, n_cluster, n_eofs))
    cluster_labels = np.zeros(shape=(n_init, n_obs))
    for i in range(n_init):
        km = KMeans(n_clusters=n_cluster).fit(data)
        cluster_centroids[i, :, :] = km.cluster_centers_
        cluster_labels[i, :] = km.labels_
    return cluster_centroids, cluster_labels

@jit
def calc_classifiability(P, Q) -> float:
    """
    Implement the Michaelangeli (1995) Classifiability Index

    Please note that the variable naming in this function does not follow
    standard python syntax. Instead, it follows the notation used in the
    Michelangeli (1995) paper, which should make it easier to relate the code
    to the text. If the syntax for this function becomes too confusing, we can
    update it.

    Parameters
    ----------
    P : ``np.ndarray``
        one cluster centroid
    Q : ``np.ndarray``
        another cluster centroid
    """
    k = P.shape[0]
    Aij = np.ones([k, k])
    for i in range(k):
        for j in range(k):
            Aij[i, j] = np.corrcoef(P[i, :], Q[j, :])[0, 1] # pearson correl
    Aprime = Aij.max(axis=0)
    ci = Aprime.min()
    return ci

@jit
def matrix_classifiability(cluster_centroids) -> float, int:
    """
    Compute the classifiability of a set of centroids

    Parameters
    ----------
    cluster_centroids : ``np.ndarray``
        array of shape (n_init, n_cluster, n_eofs) with all cluster centroids
    """
    (n_init, n_cluster, n_eofs) = cluster_centroids.shape
    pairwise_classifiability = np.ones([n_init, n_init])
    for i in range(0, nsim):
        for j in range(0, nsim):
            if i == j:
                pairwise_classifiability[i, j] = np.nan
            else:
                pairwise_classifiability[i, j] = calc_classifiability(
                    P=cluster_centroids[i, :, :],
                    Q=cluster_centroids[j, :, :]
                )
    classifiability = np.nanmean(pairwise_classifiability)
    best_part = np.argmax(pairwise_classifiability)
    return classifiability, best_part

def resort_labels(old_labels: np.ndarray) -> np.ndarray:
    """
    Re-sort cluster labels

    Takes in x, a vector of cluster labels, and re-orders them so that
    the lowest number is the most common, and the highest number
    the least common

    Parameters
    ----------
    old_labels : ``np.ndarray``
        The old labels
    Returns
    -------
    new_labels : ``np.ndarray``
        the new cluster labels, ranked by frequency of occurrence
    """
    old_labels = np.int_(old_labels) # coerce to integer
    labels_from = np.unique(old_labels).argsort()
    counts = np.array([np.sum(old_labels == xi) for xi in labels_from])
    orders = counts.argsort()[::-1].argsort()
    new_labels = orders[labels_from[old_labels]] + 1
    return new_labels

class KMeansClassifiabilityIndex(ClusteringAlgorithm):
    """
    Cluster based on the k-means algorithm, and choose the best set of clusters
    using the classifiability index of Michelangeli (1995)
    """
    def __init__(self, data: np.ndarray, n_cluster: int, n_init: int) -> None:
        """
        Parameters
        ----------
        data : ``np.ndarray``
            the input data for clustering; should only be applied once any
            desired dimension reduction has been applied
        n_cluster : ``int``
            the number of clusters to fit
        n_init : ``int``
            the number of simulations to run
        """
        super().__init__()
        self.param.update({"n_cluster": n_cluster, "n_init": n_init})

    def run(self) -> None:
        raise NotImplementedError