#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.pyrf.ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

NDArrayFloats = NDArray[Union[np.float32, np.float64]]


def trace(inp: DataArray) -> DataArray:
    r"""Computes trace of the time series of 2nd order tensors.

    Parameters
    ----------
    inp : DataArray
        Time series of the input 2nd order tensor.

    Returns
    -------
    DataArray
        Time series of the trace of the input tensor.

    Raises
    ------
    TypeError
        If inp is not a xarray.DataArray.
    ValueError
        If inp is not a time series of a tensor.

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
    if not isinstance(inp, xr.DataArray):
        raise TypeError("inp must be a xarray.DataArray")

    # Check that inp is a tensor
    if inp.ndim != 3 or inp.shape[1] != 3 or inp.shape[2] != 3:
        raise ValueError("inp must be a time series of a tensor")

    # Get diagonal elements
    inp_xx: NDArrayFloats = inp.data[:, 0, 0]
    inp_yy: NDArrayFloats = inp.data[:, 1, 1]
    inp_zz: NDArrayFloats = inp.data[:, 2, 2]

    # Compute trace
    out_data = inp_xx + inp_yy + inp_zz

    # Construct time series
    out = ts_scalar(inp.time.data, out_data, inp.attrs)

    return out
