#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contributed by Apostolos Kolokotronis."""

# 3rd party imports
import numpy as np

# Local imports
from ._time import _seconds
from ._units import _attrs, _ratio, _units


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
        time = _seconds(time, time[0])
    else:
        time = time.astype(np.float64)

        if t_units.lower() == "ns":
            time = time * 1e-9

    # NaN at the edges, where the window or stencil is not complete
    n_pts = len(data)
    derivative = np.full(data.shape, np.nan)

    def _at(values, k, half):
        # values[i + k] for the samples i with a complete stencil of half width
        # half (one slice instead of a loop over the samples)
        return values[half + k : n_pts - half + k]

    def _dt(half):
        # Time across the stencil, broadcast on the components
        d_t = _at(time, half, half) - _at(time, -half, half)
        return d_t.reshape((-1,) + (1,) * (data.ndim - 1))

    if method == "window":
        half = window_size // 2

        if n_pts > 2 * half:
            # Difference between the ends of the window
            derivative[half : n_pts - half] = (
                _at(data, half, half) - _at(data, -half, half)
            ) / _dt(half)
    elif method == "5ps":
        if n_pts > 4:
            # The centre sample has no weight: a NaN there no longer gives NaN
            derivative[2:-2] = (
                1 / 12 * _at(data, -2, 2)
                - 2 / 3 * _at(data, -1, 2)
                + 2 / 3 * _at(data, 1, 2)
                - 1 / 12 * _at(data, 2, 2)
            ) / (_dt(2) * 0.25)
    elif n_pts > 8:  # "9ps"
        derivative[4:-4] = (
            1 / 280 * _at(data, -4, 4)
            - 4 / 105 * _at(data, -3, 4)
            + 1 / 5 * _at(data, -2, 4)
            - 4 / 5 * _at(data, -1, 4)
            + 4 / 5 * _at(data, 1, 4)
            - 1 / 5 * _at(data, 2, 4)
            + 4 / 105 * _at(data, 3, 4)
            - 1 / 280 * _at(data, 4, 4)
        ) / (_dt(4) * 1 / 8)

    out = time_series.copy(data=derivative)
    # Attributes of the input, its units per s (they were replaced)
    out.attrs = _attrs(time_series.attrs, _ratio(_units(time_series), "s"))
    out.attrs["TENSOR_ORDER"] = data.ndim - 1

    return out
