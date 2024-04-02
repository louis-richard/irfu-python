#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from xarray.core.dataarray import DataArray

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def normalize(inp: DataArray) -> DataArray:
    r"""Normalizes the input field.

    Parameters
    ----------
    inp : DataArray
        Time series of the input field.

    Returns
    -------
    DataArray
        Time series of the normalized input field.

    Raises
    ------
    TypeError
        If input is not a DataArray.
    ValueError
        If input is not a 2D DataArray.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field

    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)

    Compute the normalized magnetic field

    >>> b = pyrf.normalize(b_xyz)

    """

    if not isinstance(inp, DataArray):
        raise TypeError("Input must be a DataArray")

    if inp.data.ndim == 2:
        out_data = inp.data / np.linalg.norm(inp.data, axis=1, keepdims=True)
        out = inp.copy(data=out_data)
    else:
        raise ValueError("Input must be a 2D DataArray")

    return out
