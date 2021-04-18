#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""gse2gsm.py
@author: Louis Richard
"""

from .cotrans import \
    cotrans


def gse2gsm(inp, flag="gse>gsm"):
    """Converts GSE to GSM.

    Parameters
    ----------
    inp : xarray.DataArray or ndarray
        Time series of the input in GSE (GSM) coordinates.
        If ndarray first column is time in unix format.

    flag : str
        Flag for conversion direction. "gse>gsm" or "gsm>gse".
        Default is "gse>gsm"

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

    out = cotrans(inp, flag);

    return out