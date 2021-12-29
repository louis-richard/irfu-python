#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pdb

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.12"
__status__ = "Prototype"


def hpca_spin_sum(inp, saz, method: str = "mean"):
    r"""Sum or average teh Hot Plasma Composition Analyser (HPCA) data over
    each spin.

    Parameters
    ----------
    inp : xarray.DataArray
        Ion PSD or flux; [nt, npo16, ner63], looking direction
    saz : xarray.DataArray
        Start azimuthal spin indices.
    method : {"mean", "sum"}, Optional
        Method either "sum" or "mean". Default is "mean"

    Returns
    -------
    out : xarray.DataArray
        Distribution averaged over each spin.

    """

    az_times, start_az = [saz.time.data, saz.data]

    spin_starts = np.squeeze(np.argwhere(start_az == 0))

    out_data = []
    for i, spin in enumerate(spin_starts[:-1]):
        if method == "mean":
            out_data.append(inp[spin:spin_starts[i + 1]].mean(dim='time').data)
        elif method == "sum":
            out_data.append(inp[spin:spin_starts[i + 1]].sum(dim='time').data)
        else:
            raise ValueError("Invalid method")

    out_time = np.stack([t for t in az_times[spin_starts[:-1]]])
    out_data = np.stack(out_data)
    coords = [inp.coords[k].data for k in inp.dims[1:]]
    coords = [out_time, *coords]
    dims = list(inp.dims)

    out = xr.DataArray(out_data, coords=coords, dims=dims, attrs=inp.attrs)

    return out
