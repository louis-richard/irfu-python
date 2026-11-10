#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import xarray as xr

# Local imports
from .cotrans import cotrans

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def gse2gsm(inp, flag: str = "gse>gsm"):
    r"""Converts GSE to GSM.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input in GSE (GSM) coordinates.
    flag : {"gse>gsm", "gsm>gse"}, Optional
        Flag for conversion direction (case-insensitive). Default is "gse>gsm".

    Returns
    -------
    out : xarray.DataArray
        Time series of the input in GSM (GSE) coordinates.

    Raises
    ------
    TypeError
        If inp is not a xarray.DataArray.
    ValueError
        If inp is not a vector, or flag not "gse>gsm" or "gsm>gse".

    See Also
    --------
    pyrfu.pyrf.cotrans

    """

    if not isinstance(inp, xr.DataArray):
        raise TypeError("inp must be a xarray.DataArray")

    if inp.ndim != 2 or inp.shape[1] != 3:
        raise ValueError("inp must be a vector")

    if not isinstance(flag, str) or flag.lower() not in ["gse>gsm", "gsm>gse"]:
        raise ValueError(f'flag must be "gse>gsm" or "gsm>gse", not {flag!r}')

    out = cotrans(inp, flag.lower())

    return out
