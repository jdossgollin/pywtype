"""
Reduce the dimension of the input object
"""

from sklearn.decomposition import PCA

from .dataset import Dataset

class DimensionReducer:
    """
    A generic class for dimension reduction
    """
    def __init__(self, data: Dataset) -> None:
        """
        Generic class for dimension reduction
        """
        raise NotImplementedError

    def fit(self) -> Dataset:
        """
        A generic class for fitting
        """