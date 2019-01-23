"""
The wtype object
"""

from typing import Tuple
import xarray as xr

class WeatherType:
    """
    The object for building a weather typing model
    """
    def __init__(
        self, data: xr.DataArray, space_name: Tuple[str],
        time_name: str = "time") -> None:
        """
        Parameters
        ----------
        data : ``xr.DataArray``, required
            the input space-time field
        time_name : ``str``, optional, (default = "time")
            the name of the dimension which indexes time.
        space_name : ``Tuple[str]``,  required
            a tuple containing the name of all spatial dimensions
        """
        data = data.rename({time_name, "time"}).stack(space=space_name)
        assert data.ndim == 2, "data must have two dimensions exactly"
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
