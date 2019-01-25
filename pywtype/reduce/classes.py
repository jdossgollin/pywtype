"""
Reduce the dimension of the input object
"""

import numpy as np
from sklearn.decomposition import PCA
from Typing import Dict

class DimensionReducer:
    """
    A generic class for dimension reduction
    """
    def __init__(self) -> None:
        """
        Generic class for dimension reduction
        """
        self.param: Dict = {}

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        A generic class for fitting, must be implemented by each child class

        Parameters
        ----------
        data : ``np.ndarray``
            the input data to use
        """

    def get_param(self) -> Dict:
        """
        Get all the parameters
        """
        return self.param

class FixedCountPCAReducer(DimensionReducer):
    """
    Reduce the dimension of the field using PCA with a pre-defined number
    of EOFs

    Parameters
    ----------
    n_eof : int
        the number of EOFs (PCs) to retain
    """
    def __init__(self, n_eof: int) -> None:
        """
        Initialize the class
        """
        super().__init__()
        self.param.update({"n_eof": n_eof})

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        A generic class for fitting, must be implemented by each child class

        Parameters
        ----------
        data : ``np.ndarray``
            the data set to use
        """
        pca = PCA(n_components=n_eof).fit(data)
        pc_ts = pca.transform(data)
        return pc_ts
