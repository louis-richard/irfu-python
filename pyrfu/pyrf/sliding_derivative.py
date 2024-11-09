#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from .ts_scalar import ts_scalar

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

    Parameters:
    ----------
    time_series : xarray.DataArray
        The time series data for which the derivative is to be calculated.
    t_units: str, optional, default: "ns"
        The units of the time coordinate in time_series. If t_units is "ns" then the
        time is converted to seconds.
    window_size : int, optional, default: 3
        The number of data points in each sliding window. It should be an odd integer
        to ensure a symmetric window around the central point for central differences.

    Returns:
    -------
    derivative : xarray.DataArray
        An array containing the sliding derivative for each point in the time series.
        The length of this array matches the length of `time_series`. For points near
        the boundaries (where a full window cannot be formed), the result will be NaN.

    Notes:
    ------
    - The derivative is approximated using central differences for points that can
    accommodate the window size. For edge points, the output will contain NaN.
    - The function assumes `time_steps` are evenly spaced but works for irregular time
    steps as well by calculating the actual time difference between the start and end
    of the window.

    """

    data = time_series.data
    time = time_series.time.astype(np.float64).data

    assert t_units.lower() in ["ns", "s"], "convert time to ns or s."

    if t_units.lower() == "ns" or time.data.dtype == "<M8[ns]":
        time = time * 1e-9

    assert method.lower() in [
        "window",
        "5ps",
        "9ps",
    ], "this method has not been implemented."

    half_window = window_size // 2

    # Fill the output with NaN for edge cases
    derivative = np.full(len(data), np.nan)
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

    # time_dt64 = ts_time(time).data
    out = ts_scalar(
        time_series.time.data,
        derivative,
    )

    return out
