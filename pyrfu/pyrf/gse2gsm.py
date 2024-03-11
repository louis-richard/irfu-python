#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import xarray as xr

# Local imports
from .cotrans import cotrans

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def gse2gsm(inp, flag: str = "gse>gsm"):
    r"""Converts GSE to GSM.

    Parameters
    ----------
    inp : xarray.DataArray or ndarray
        Time series of the input in GSE (GSM) coordinates.
        If ndarray first column is time in unix format.
    flag : {"gse>gsm", "gsm>gse"}, Optional
        Flag for conversion direction. Default is "gse>gsm"

    Returns
    -------
    out : xarray.DataArray or ndarray
        Time series of the input in GSM (GSE) coordinates.
        If ndarray first column is time in unix format.

    See also
    --------
    pyrfu.pyrf.geocentric_coordinate_transformation

    """

    assert isinstance(inp, xr.DataArray), "inp must be a xarray.DataArray"
    assert inp.ndim == 2 and inp.shape[1] == 3, "inp must be a vector"

    message = "flag must be a string gse>gsm or gsm>gse"
    assert isinstance(flag, str) and flag.lower() in ["gse>gsm", "gsm>gse"], message

    out = cotrans(inp, flag)

    return out
