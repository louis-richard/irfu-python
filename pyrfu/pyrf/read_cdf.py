#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import pycdfpp
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def read_cdf(path: str) -> dict:
    r"""Reads a .cdf file and returns a dictionary with the fields contained in
    the file.

    Parameters
    ----------
    path : str
        Path to the .cdf file.

    Returns
    -------
    dict
        Hash table with fields contained in the .cdf file.

    """

    # Initialize output dictionary
    out_dict = {}

    # Load file
    file = pycdfpp.load(path)

    # Get keys (a.k.a zvariables) from file
    keys = list(map(lambda x: x[0], file.items()))

    for key in keys:
        # Get data and coordinates
        data = np.squeeze(file[key].values)
        coords = [np.arange(dim_size) for dim_size in data.shape]

        # Construct xarray DataArray
        out_dict[key.lower()] = xr.DataArray(data, coords=coords)

    return out_dict
