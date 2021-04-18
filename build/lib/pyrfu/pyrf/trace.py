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

"""trace.py
@author: Louis Richard
"""

import xarray as xr


def trace(inp):
    """Computes trace of the time series of 2nd order tensors.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the input 2nd order tensor.

    Returns
    -------
    out : xarray.DataArray
        Time series of the trace of the input tensor.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field and ion temperature

    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> t_xyz_i = mms.get_data("Ti_gse_fpi_fast_l2", tint, mms_id)

    Rotate to ion temperature tensor to field aligned coordinates

    >>> t_xyzfac_i = mms.rotate_tensor(t_xyz_i, "fac", b_xyz, "pp")

    Compute scalar temperature

    >>> t_i = pyrf.trace(t_xyzfac_i)

    """

    inp_data = inp.data
    out_data = inp_data[:, 0, 0] + inp_data[:, 1, 1] + inp_data[:, 2, 2]

    # Attributes
    attrs = inp.attrs

    # Change tensor order from 2 (matrix) to 0 (scalar)
    attrs["TENSOR_ORDER"] = 0

    out = xr.DataArray(out_data, coords=[inp.time.data], dims=["time"],
                       attrs=attrs)

    return out
