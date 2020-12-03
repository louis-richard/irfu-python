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

import numpy as np
import xarray as xr


def ts_tensor_xyz(t=None, data=None, attrs=None):
    """Create a time series containing a 2nd order tensor.

    Parameters
    ----------
    t : numpy.ndarray
        Array of times.

    data : numpy.ndarray
        Data corresponding to the time list.

    attrs : dict
        Attributes of the data list.

    Returns
    -------
    out : xarray.DataArray
        2nd order tensor time series.

    """

    assert t is not None and isinstance(t, np.ndarray)
    assert data is not None and isinstance(data, np.ndarray)
    assert data.ndim == 3 and data.shape[1:] == (3, 3)

    # Check inputs are not empty
    if t is None:
        raise ValueError("ts_tensor_xyz requires at least two inputs")

    if data is None:
        raise ValueError("ts_tensor_xyz requires at least two inputs")

    # Check inputs are numpy arrays
    if len(t) != len(data):
        raise IndexError("Time and data must have the same length")

    flag_attrs = True

    if attrs is None:
        flag_attrs = False

    out = xr.DataArray(data, coords=[t[:], ["x", "y", "z"], ["x", "y", "z"]],
                       dims=["time", "comp_h", "comp_v"])

    if flag_attrs:
        out.attrs = attrs

        out.attrs["TENSOR_ORDER"] = 2
    else:
        out.attrs["TENSOR_ORDER"] = 2

    return out
