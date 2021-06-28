#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Local imports
from .cotrans import cotrans

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
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

    assert flag in ["gse>gsm", "gsm>gse"], "invalid flag"

    out = cotrans(inp, flag)

    return out
