#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

from cdflib import cdfread

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def get_variable(file_path, cdf_name):
    r"""Reads field named cdf_name in file and convert to DataArray.

    Parameters
    ----------
    file_path : str
        Path of the cdf file.
    cdf_name : str
        Name of the target variable in the cdf file.

    Returns
    -------
    out : xarray.DataArray
        Target variable.

    """

    with cdfread.CDF(file_path) as file:
        var_data = file.varget(cdf_name)
        var_atts = file.varattsget(cdf_name)

    out = xr.DataArray(var_data, coords=[np.arange(len(var_data))], dims=["x"],
                       attrs=var_atts)

    return out
