#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Optional

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def integrate(inp: DataArray, time_step: Optional[float] = None) -> DataArray:
    r"""Integrate time series, as irfu-matlab irf_integrate.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the variable to integrate (scalar, vector or tensor).
    time_step : float, Optional
        Time step in seconds: the steps larger than 3 time_step are data gaps,
        not integrated. Default is the smallest time step but one (in case
        of a faulty time tag), as irf_integrate.

    Returns
    -------
    out : xarray.DataArray
        Time series of the time integral of inp, zero at the first time, with
        the attributes of inp (the units times s).

    Raises
    ------
    TypeError
        If inp is not a xarray.DataArray.

    Notes
    -----
    The integral is the cumulative sum of x(t_i) (t_i - t_{i - 1}), as
    irf_integrate, with the times in seconds from the first one (exact to
    the ns). A step over a data gap counts as zero, and a NaN sample is
    NaN in the output and is not integrated (nor is its time step).

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    >>> tint = ["2015-12-14T01:17:40.200", "2015-12-14T01:17:41.500"]
    >>> e_xyz = mms.get_data("e_gse_edp_brst_l2", tint, 1)
    >>> int_e_dt = pyrf.integrate(e_xyz)

    """

    if not isinstance(inp, DataArray):
        raise TypeError("inp must be a xarray.DataArray")

    # Times in seconds from the first one, from the integer ns (float64
    # seconds since 1970 would be rounded to 256 ns)
    time_ns = inp.time.data.astype("datetime64[ns]").astype(np.int64)
    time = (time_ns - time_ns[0]) * 1e-9
    delta_t = np.diff(time, prepend=time[0])

    if time_step is None:
        # The smallest step but one (the only one for two samples)
        steps = np.sort(np.diff(time))
        time_step = steps[min(1, len(steps) - 1)] if len(steps) else 0.0

    delta_t[delta_t > 3 * time_step] = 0.0

    # Cumulative sum over the finite samples of each component
    data = np.asarray(inp.data, dtype=np.float64).reshape(len(time), -1)
    is_nan = np.isnan(data)
    out_data = np.cumsum(np.where(is_nan, 0.0, data * delta_t[:, None]), axis=0)
    out_data[is_nan] = np.nan

    attrs = dict(inp.attrs)

    if "UNITS" in attrs:
        attrs["UNITS"] = f"({attrs['UNITS']})*s"

    out = xr.DataArray(
        out_data.reshape(inp.shape), coords=inp.coords, dims=inp.dims, attrs=attrs
    )

    return out
