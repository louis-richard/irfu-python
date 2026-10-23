#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def hpca_spin_sum(inp, saz, method: str = "mean"):
    r"""Sum or average the Hot Plasma Composition Analyser (HPCA) data over
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
        Distribution averaged (or summed) over each spin, at the time of its
        start.

    Raises
    ------
    ValueError
        If method is not "mean" or "sum", or if saz has fewer than two spin
        starts (no complete spin).

    Notes
    -----
    A spin runs from one start (start azimuth index 0) to the next: the
    samples after the last start, a spin not known to be complete, are not
    used, as pyspedas ``mms_hpca_spin_sum`` (IDL SPEDAS gives a spin of zeros
    there).

    """

    if method not in ("mean", "sum"):
        raise ValueError(f"Invalid method {method!r}, use mean or sum")

    az_times, start_az = [saz.time.data, saz.data]

    # Indices of the spin starts (one start gave a 0-d array: IndexError)
    spin_starts = np.flatnonzero(start_az == 0)

    if len(spin_starts) < 2:
        raise ValueError(
            "saz must have at least two spin starts (start azimuth index 0): "
            "no complete spin"
        )

    out_data = []
    for i, spin in enumerate(spin_starts[:-1]):
        spin_data = inp[spin : spin_starts[i + 1]]

        if method == "mean":
            out_data.append(spin_data.mean(dim="time").data)
        else:
            out_data.append(spin_data.sum(dim="time").data)

    out_time = np.stack(az_times[spin_starts[:-1]])
    out_data = np.stack(out_data)
    coords = [inp.coords[k].data for k in inp.dims[1:]]
    coords = [out_time, *coords]
    dims = list(inp.dims)

    out = xr.DataArray(out_data, coords=coords, dims=dims, attrs=inp.attrs)

    return out
