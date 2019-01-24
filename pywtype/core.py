"""
Define the base class here
"""

from typing import Tuple
import xarray as xr

class WeatherType:
    """
    The entire model goes here
    """
    def __init__(
            self,
            data: xr.DataArray,
            time_dim: str, space_dims: Tuple[str],
            project, cluster
        ) -> None:
        """
        Set up the Dataset

        Parameters
        ----------
        data : xr.DataArray object
            The input data set to use. May have multiple dimensions.
            At present, must be a DataArray (i.e., not a Dataset with more than
            one variable).
        time_dim : str
            The name of the dimension which indexes time
        space_dims: Tuple[str]
            A Tuple of strings giving the name of all dimensions which
            index space
        """
        data = data.rename(time_dim, "time").stack(space=space_dims)
        assert data.ndims == 2, "must have two dimensions"
        data = data.resample(time="1D").mean(dim="time")
        self.data = data
        self.time_dim = time_dim
        self.space_dims = space_dims
