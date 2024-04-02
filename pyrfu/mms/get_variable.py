#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from pycdfpp import _pycdfpp, load, to_datetime64
from xarray.core.dataarray import DataArray

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def _pycdfpp_attributes_to_dict(attributes):
    attributes_dict = {}

    for k in attributes:
        tmp = [attributes[k][i] for i in range(len(attributes[k]))]

        if np.size(tmp) == 1:
            if isinstance(tmp[0], (list, np.ndarray)) and isinstance(
                tmp[0][0], _pycdfpp.tt2000_t
            ):
                attributes_dict[k] = to_datetime64(tmp[0][0])
            else:
                attributes_dict[k] = tmp[0]
        else:
            attributes_dict[k] = tmp[:]

    return attributes_dict


def get_variable(file_path: str, cdf_name: str) -> DataArray:
    r"""Read field named cdf_name in file and convert to DataArray.

    Parameters
    ----------
    file_path : str
        Path of the cdf file.
    cdf_name : str
        Name of the target variable in the cdf file.

    Returns
    -------
    out : DataArray
        Target variable.

    """
    # Load file
    file = load(file_path)

    var_data = file[cdf_name].values
    var_attributes = _pycdfpp_attributes_to_dict(file[cdf_name].attributes)

    out = xr.DataArray(
        var_data,
        coords=[np.arange(len(var_data))],
        dims=["x"],
        attrs=var_attributes,
    )

    return out
