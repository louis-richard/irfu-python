#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
import xarray as xr
from pycdfpp import _pycdfpp, load, to_datetime64
from xarray.core.dataarray import DataArray


def _pycdfpp_attributes_to_dict(attributes):
    r"""Attributes of a CDF file or variable as a dictionary: the value of the
    attributes of one entry (as a list for several values, datetime64 for
    times), the list of the entries of the others (global attributes)."""
    attributes_dict = {}

    for k in attributes:
        entries = [attributes[k][i] for i in range(len(attributes[k]))]

        if len(entries) != 1:
            attributes_dict[k] = entries
            continue

        value = entries[0]

        # (an attribute of several values was a list of one list)
        if isinstance(value, (list, np.ndarray)) and isinstance(
            value[0], _pycdfpp.tt2000_t
        ):
            attributes_dict[k] = to_datetime64(value[0] if len(value) == 1 else value)
        else:
            attributes_dict[k] = value

    return attributes_dict


def get_variable(file_path: Union[str, bytes], cdf_name: str) -> DataArray:
    r"""Read field named cdf_name in file and convert to DataArray.

    Parameters
    ----------
    file_path : str or bytes
        Path of the cdf file, or its content.
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

    # Dimensions "x" (as for 1-D variables), then "y", "z" for N-D variables
    dims = [["x", "y", "z"][i] if i < 3 else f"dim_{i}" for i in range(var_data.ndim)]

    out = xr.DataArray(
        var_data,
        coords=[np.arange(n) for n in var_data.shape],
        dims=dims,
        attrs=var_attributes,
    )

    return out
