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

"""spin_hpca_avg.py
@author: Louis Richard
"""

import numpy as np
import xarray as xr


def spin_hpca_avg(vdf, saz, method: str = "mean"):
    r"""
    Sum or average teh Hot Plasma Composition Analyser (HPCA) data over each
    spin.

    Parameters
    ----------
    vdf : xarray.DataArray
        Ion PSD or flux; [nt, npo16, ner63], looking direction

    saz : xarray.DataArray
        Start azimuthal spin indices.

    method : str, optional
        Method either "sum" or "mean". Default is "mean"

    Returns
    -------
    out : xarray.DataArray
        Distribution averaged over each spin.

    """

    az_times, start_az = [saz.time.data, saz.data]

    spin_starts = np.argwhere(start_az == 0)

    out_data = []
    for i, spin_start in enumerate(spin_starts[:-1]):
        if method == "mean":
            out_data.append(
                vdf[spin_start[0]:spin_starts[i + 1][0]].mean(dim='time').data)
        elif method == "sum":
            out_data.append(
                vdf[spin_start[0]:spin_starts[i + 1][0]].sum(dim='time').data)
        else:
            raise ValueError("Invalid method")

    out_time = np.stack([t[0] for t in az_times[spin_starts[:-1]]])
    out_data = np.stack(out_data)
    coords = [vdf.coords[k].data for k in vdf.dims[1:]]
    coords = [out_time, *coords]
    dims = list(vdf.dims)

    out = xr.DataArray(out_data, coords=coords, dims=dims, attrs=vdf.attrs)

    return out
