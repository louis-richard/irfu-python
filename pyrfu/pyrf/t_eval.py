#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np


def t_eval(inp, times):
    r"""Evaluates the input time series at the target time.

    There is no interpolation: each target time takes the first sample of the
    input at or after it (the first sample for the times before the input).
    Use :func:`pyrfu.pyrf.resample` to interpolate.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input to evaluate, of any shape.
    times : ndarray
        Times at which the input will be evaluated.

    Returns
    -------
    out : xarray.DataArray
        Time series of the input at times t.

    Raises
    ------
    IndexError
        If a target time is after the last sample of the input.

    """

    # First sample at or after each target time (a loop with bisect)
    times = np.asarray(times)
    idx = np.searchsorted(inp.time.data, times, side="left")

    if np.any(idx >= len(inp.time)):
        raise IndexError("a target time is after the last sample of inp")

    # Any shape, with the coordinates and attributes of inp (the second
    # dimension had to be "comp", e.g. not the FGM represent_vec_tot)
    out = inp.isel(time=idx).assign_coords(time=times)
    out.attrs = dict(inp.attrs)

    return out
