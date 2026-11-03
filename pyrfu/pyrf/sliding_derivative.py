#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from ._units import _attrs, _ratio, _units

__author__ = "Apostolos Kolokotronis"
__email__ = "apostolos.kolokotronis@irf.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def sliding_derivative(
    time_series, t_units: str = "ns", window_size=3, method: str = "window"
):
    """
    Compute the sliding time derivative of a time series using central differences.

    This function calculates the rate of change of a time series at each point by
    computing the derivative within a sliding window of a specified size. The derivative
    at each point is estimated using the first and last points in the window (central
    difference).

    Parameters
    ----------
    time_series : xarray.DataArray
        The time series data for which the derivative is to be calculated (scalar,
        vector or tensor).
    t_units : {"ns", "s"}, optional, default: "ns"
        The units of the time coordinate in time_series when it is not
        ``datetime64`` (``datetime64`` times are always converted to seconds).
    window_size : int, optional, default: 3
        The number of data points in each sliding window (method "window"). It
        should be an odd integer to ensure a symmetric window around the central
        point for central differences.
    method : {"window", "5ps", "9ps"}, optional, default: "window"
        Difference between the ends of the sliding window, or 5-point or 9-point
        central-difference stencil (assumes evenly spaced times). Case-insensitive.

    Returns
    -------
    derivative : xarray.DataArray
        An array containing the sliding derivative for each point in the time series,
        per second, with the shape and coordinates of `time_series`. For points near
        the boundaries (where a full window cannot be formed), the result will be NaN.

    Raises
    ------
    ValueError
        If t_units or method is not one of the above.

    Notes
    -----
    - The derivative is approximated using central differences for points that can
      accommodate the window size. For edge points, the output will contain NaN.
    - The function assumes `time_steps` are evenly spaced but works for irregular time
      steps as well by calculating the actual time difference between the start and
      end of the window.

    """

    if t_units.lower() not in ["ns", "s"]:
        raise ValueError(f"t_units must be 'ns' or 's', not {t_units!r}")

    method = method.lower()

    if method not in ["window", "5ps", "9ps"]:
        raise ValueError(f"method must be 'window', '5ps' or '9ps', not {method!r}")

    data = time_series.data
    time = time_series.time.data

    if np.issubdtype(time.dtype, np.datetime64):
        # Seconds from the first sample (epochs in ns as float64 are rounded to
        # 256 ns, i.e. 0.1 % of the time step of 8192 Hz data)
        time = (time - time[0]).astype("timedelta64[ns]").astype(np.float64) * 1e-9
    else:
        time = time.astype(np.float64)

        if t_units.lower() == "ns":
            time = time * 1e-9

    # Fill the output with NaN for edge cases
    derivative = np.full(data.shape, np.nan)
    if method == "window":

        half_window = window_size // 2

        # Iterate over each window in the time series
        for i in range(half_window, len(data) - half_window):
            # Get the window of values and time
            values_window = data[i - half_window : i + half_window + 1]
            time_window = time[i - half_window : i + half_window + 1]

            # Compute finite differences (central difference for the middle point)
            derivative[i] = (values_window[-1] - values_window[0]) / (
                time_window[-1] - time_window[0]
            )
    elif method == "5ps":

        for i in range(2, len(data) - 2):

            derivative[i] = (
                1 / 12 * data[i - 2]
                - 2 / 3 * data[i - 1]
                + 0 * data[i]
                + 2 / 3 * data[i + 1]
                - 1 / 12 * data[i + 2]
            ) / ((time[i + 2] - time[i - 2]) * 0.25)
    elif method == "9ps":

        for i in range(4, len(data) - 4):

            derivative[i] = (
                1 / 280 * data[i - 4]
                - 4 / 105 * data[i - 3]
                + 1 / 5 * data[i - 2]
                - 4 / 5 * data[i - 1]
                + 0 * data[i]
                + 4 / 5 * data[i + 1]
                - 1 / 5 * data[i + 2]
                + 4 / 105 * data[i + 3]
                - 1 / 280 * data[i + 4]
            ) / ((time[i + 4] - time[i - 4]) * 1 / 8)

    out = time_series.copy(data=derivative)
    # Attributes of the input, its units per s (they were replaced)
    out.attrs = _attrs(time_series.attrs, _ratio(_units(time_series), "s"))
    out.attrs["TENSOR_ORDER"] = data.ndim - 1

    return out
