#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def calc_dt(inp: Union[Dataset, DataArray]) -> float:
    r"""Compute time step of the input time series.

    Parameters
    ----------
    inp : DataArray or Dataset
        Time series of the input variable.

    Returns
    -------
    float
        Time step in seconds.

    """
    # Check input type
    if not isinstance(inp, (Dataset, DataArray)):
        raise TypeError("Input must be a time series")

    return float(np.median(np.diff(inp.time.data)).astype(np.float64) * 1e-9)
