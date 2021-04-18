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

"""ts_append.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr


def ts_append(inp1, inp2):
    """Concatenate two time series along the time axis.

    Parameters
    ----------
    inp1 : xarray.DataArray
        Time series of the first input (early times).

    inp2 : xarray.DataArray
        Time series of the second input (late times).

    Returns
    -------
    out : xarray.DataArray
        Concatenated time series.

    Notes
    -----
    The time series must be in the correct time order.

    """

    if inp1 is None:
        return inp2

    out_data = {}

    if inp1.data.ndim != 1:
        out_data["data"] = np.vstack([inp1, inp2])

    else:
        out_data["data"] = np.hstack([inp1, inp2])

    out_data["attrs"] = {}

    for k in inp1.attrs:
        if isinstance(inp1.attrs[k], np.ndarray):
            out_data["attrs"][k] = np.hstack([inp1.attrs[k], inp2.attrs[k]])

        else:
            out_data["attrs"][k] = inp1.attrs[k]

    depends = [{} for _ in range(len(inp1.dims))]

    for i, dim in enumerate(inp1.dims):
        if i == 0 or dim == "time":
            depends[i]["data"] = np.hstack([inp1[dim].data, inp2[dim].data])

            # add attributes
            depends[i]["attrs"] = {}

            for k in inp1[dim].attrs:
                # if attrs is array time append
                if isinstance(inp1[dim].attrs[k], np.ndarray):
                    depends[i]["attrs"][k] = np.hstack([inp1[dim].attrs[k],
                                                        inp2[dim].attrs[k]])

                else:
                    depends[i]["attrs"][k] = inp1[dim].attrs[k]

        else:
            # Use values of other coordinates of inp1 assuming equal to inp2
            depends[i]["data"] = inp1[dim].data

            # add attributes
            depends[i]["attrs"] = {}

            for k in inp1[dim].attrs:
                depends[i]["attrs"][k] = inp1[dim].attrs[k]

    # Create DataArray
    out = xr.DataArray(out_data["data"],
                       coords=[depend["data"] for depend in depends],
                       dims=inp1.dims, attrs=out_data["attrs"])

    for i, dim in enumerate(out.dims):
        out[dim].attrs = depends[i]["attrs"]

    return out
