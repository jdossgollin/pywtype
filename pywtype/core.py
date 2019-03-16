"""
Define the base class here
"""

from typing import Tuple
import pandas as pd
import xarray as xr

from .reduce.classes ipmort DimensionReducer
from .cluster.classes import ClusteringAlgorithm

class WeatherTypeModel:
    """
    The entire model goes here
    """
    def __init__(
            self,
            data: xr.DataArray,
            time_dim: str, space_dims: Tuple[str],
            reducer: DimensionReducer,
            cluster: ClusteringAlgorithm
        ) -> None:
        """
        Set up the Dataset

        Parameters
        ----------
        data : ``xr.DataArray``
            The input data set to use. May have multiple dimensions.
            At present, must be a DataArray (i.e., not a Dataset with more than
            one variable).
        time_dim : ``str``
            The name of the dimension which indexes time
        space_dims : ``Tuple[str]``
            A Tuple of strings giving the name of all dimensions which
            index space
        reducer : ``DimensionReducer``
            A ``DimensionReducer`` object for reducing the dimension of the
            input data
        cluster : ``ClusteringAlgorithm``
            A ``ClusteringAlgorithm`` object for performing clustering in the
            reduced-dimension space which returns the weather types
        """
        data = data.rename(time_dim, "time").stack(space=space_dims)
        assert data.ndims == 2, "must have two dimensions"
        data = data.resample(time="1D").mean(dim="time") # must be daily
        self.data = data
        self.time_dim = time_dim
        self.space_dims = space_dims
        self.reducer = reducer
        self.cluster = cluster
    
    def fit(self) -> None:
        """
        Fit the full weather typing algorithm
        """
        reduced_data = self.reducer.fit(self.data.values)
        self.weather_types, self.score = self.cluster.fit(reduced_data)
        self.weather_types = (
            pd.Series(self.weather_types, index=reduced_data["time"]).
            to_xarray()

        )
        
