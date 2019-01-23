"""
The wtype object
"""

import xarray as xr

class WeatherType:
    """
    The object for building a weather typing model
    """
    def __init__(self, data: xr.DataArray, time_name: str = "time") -> None:
        """
        Parameters
        ----------
        data : ``xr.DataArray``, required
            a data array which *must* be two-dimensional. It may be helpful
            to use the stack method.
        time_name : ``str``, optional, (default = "time")
            the name of the dimension which indexes time.
        """
        assert data.ndim == 2, "data must have two dimensions exactly"
        data = data.rename({time_name, "time"})
        self.data = data
        raise NotImplementedError

    def fit(self) -> None:
        """
        Fit the weather typing model
        """
        raise NotImplementedError

    def get_wtypes(self) -> None:
        """
        get the fitted weather types
        """
        raise NotImplementedError

    def plot(self) -> None:
        """
        plot the weather types
        """
