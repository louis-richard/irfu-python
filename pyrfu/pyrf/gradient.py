#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Optional

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def gradient(inp, time_step: Optional[float] = None):
    r"""Computes time derivative of the input variable.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input variable.
    time_step : float, Optional
        Time step in seconds: the steps larger than 3 time_step are data gaps,
        across which the series is not differentiated. Default is the smallest
        time step but one (in case of a faulty time tag), as irfu-matlab
        irf_gradient.

    Returns
    -------
    out : xarray.DataArray
        Time series of the time derivative of the input variable, per second.

    Notes
    -----
    Each segment between data gaps is differentiated separately, as
    irf_gradient: second-order central differences with the actual time steps
    (also for uneven sampling), and first-order differences at the ends of the
    segment (see numpy.gradient). A segment of a single sample is NaN.
    irf_gradient uses the regular time step, and its segments are off by one:
    it differentiates across the gaps at the last sample before each gap and
    the first after a gap followed by another (168 instead of about 18 for
    t^2 at t = 9 s before a 10 s gap).

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2017-07-18T13:03:34.000", "2017-07-18T13:07:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field

    >>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, mms_id)

    Time derivative of the magnetic field

    >>> db_dt = pyrf.gradient(b_xyz)

    """

    # Times in seconds from the first sample, from the exact time steps in ns (the
    # median time step was wrong across gaps and for jittered times)
    time = inp.time.data.astype("datetime64[ns]")
    time = (time - time[0]).astype(np.int64) * 1e-9
    steps = np.diff(time)

    if time_step is None:
        # The smallest step but one (the only one for two samples)
        sorted_steps = np.sort(steps)
        time_step = sorted_steps[min(1, len(steps) - 1)] if len(steps) else np.inf

    # Segments between the data gaps, differentiated separately (they were
    # differentiated across the gaps)
    bounds = np.r_[0, np.flatnonzero(steps > 3 * time_step) + 1, len(time)]
    data = np.asarray(inp.data, dtype=np.float64)
    d_inp_dt = np.full(data.shape, np.nan)

    for i_start, i_stop in zip(bounds[:-1], bounds[1:]):
        if i_stop - i_start > 1:
            d_inp_dt[i_start:i_stop] = np.gradient(
                data[i_start:i_stop], time[i_start:i_stop], axis=0
            )

    out = xr.DataArray(
        d_inp_dt,
        coords=inp.coords,
        dims=inp.dims,
        attrs=inp.attrs,
    )

    if "UNITS" in out.attrs:
        out.attrs["UNITS"] += "/s"

    return out
