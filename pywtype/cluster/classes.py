"""
Implement clustering algorithms
"""

from typing import Tuple, Dict
import numpy as np
from sklearn.preprocessing import StandardScaler
from .functions import resort_labels, matrix_classifiability, loop_kmeans

class ClusteringAlgorithm:
    """
    A generic class for implementing clustering
    """
    def __init__(self) -> None:
        """
        Set up the clustering algorithm
        """
        self.param: Dict = {}
        self.is_fitted = False
        self.score = None
        self.weather_types = None

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the model -- must be reimplemented for each child class

        Parameters
        ----------
        data : ``np.ndarray``
            the input data for clustering; should only be applied once any
            desired dimension reduction has been applied
        """

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
    
    def get_weather_types(self) -> np.ndarray:
        """
        Get the score of the model
        """
        if not self.is_fitted:
            raise ValueError((
                "Cannot return the weather types because"
                "the model has not yet been fit"
            ))
        return self.weather_types

class KMeansClassifiabilityIndex(ClusteringAlgorithm):
    """
    Cluster based on the k-means algorithm, and choose the best set of clusters
    using the classifiability index of Michelangeli (1995)
    """
    def __init__(self, n_cluster: int, n_init: int, rescale: bool = False) -> None:
        """
        Parameters
        ----------
        n_cluster : ``int``
            the number of clusters to fit
        n_init : ``int``
            the number of simulations to run
        rescale : ``bool``, default=False
            Re-scale the principal components before clustering?
            This is common practice in clustering algorithms if all columns of
            the data are equivalent. However, in weather typing it has the
            effect of treating all EOFs retained as equally important, which
            may or may not be desirable. We recommend leaving this False
            unless you have a good reason to do otherwise
        """
        super().__init__()
        self.param.update({"n_cluster": n_cluster, "n_init": n_init})

    def fit(self, data: np.ndarray) -> None:
        """
        Compute the k-means clusters

        Parameters
        ----------
        data : ``np.ndarray``
            the input data for clustering; should only be applied once any
            desired dimension reduction has been applied
        """
        if rescale:
            data = StandardScaler().fit_transform(data)
        
        cluster_centroids, cluster_labels = loop_kmeans(
            data=data, 
            n_cluster=self.param.get("n_cluster"),
            n_init=self.param.get("n_init")
        )
        classifiability, best_part = matrix_classifiability(cluster_centroids)
        weather_types = cluster_labels[best_part, :]
        weather_types = resort_labels(weather_types)
        self.score = classifiability
        self.weather_types = weather_types
        self.is_fitted = True
