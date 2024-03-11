#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import xarray as xr

# Local imports
from .ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def trace(inp):
    r"""Computes trace of the time series of 2nd order tensors.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input 2nd order tensor.

    Returns
    -------
    out : xarray.DataArray
        Time series of the trace of the input tensor.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field and ion temperature

    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> t_xyz_i = mms.get_data("Ti_gse_fpi_fast_l2", tint, mms_id)

    Rotate to ion temperature tensor to field aligned coordinates

    >>> t_xyzfac_i = mms.rotate_tensor(t_xyz_i, "fac", b_xyz, "pp")

    Compute scalar temperature

    >>> t_i = pyrf.trace(t_xyzfac_i)

    """

    # Check input type
    assert isinstance(inp, xr.DataArray), "inp must be a xarray.DataArray"

    # Check that inp is a tensor
    message = "inp must be a time series of a tensor"
    assert inp.ndim == 3 and inp.shape[1] == 3 and inp.shape[1] == 3, message

    out_data = inp.data[:, 0, 0] + inp.data[:, 1, 1] + inp.data[:, 2, 2]

    # Construct time series
    out = ts_scalar(inp.time.data, out_data, inp.attrs)

    return out
