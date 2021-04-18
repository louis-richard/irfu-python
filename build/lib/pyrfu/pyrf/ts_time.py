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

"""ts_time.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr

from astropy.time import Time


def ts_time(time, fmt="unix"):
    """Creates time line in DataArray.

    Parameters
    ----------
    time : ndarray
        Input time line.

    fmt : str
        Format of the input time line.

    Returns
    -------
    out : xarray.DataArray
        Time series of the time line.

    """

    assert isinstance(time, np.ndarray)

    time = Time(time, format=fmt).datetime64

    out = xr.DataArray(time, coords=[time], dims=["time"])

    return out
