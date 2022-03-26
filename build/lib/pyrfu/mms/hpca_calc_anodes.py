#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"

anodes_theta = np.array([123.75000, 101.25000, 78.750000, 56.250000,
                         33.750000, 11.250000, 191.25000, 213.75000,
                         236.25000, 258.75000, 281.25000, 303.75000,
                         326.25000, 348.75000, 168.75000, 146.25000])


def hpca_calc_anodes(inp, fov: list = None, method: str = "mean"):
    r"""Averages over anodes (or a given field of view) for HPCA ion data.

    Parameters
    ----------
    inp : xarray.DataArray
        Ion flux; [nt, npo16, ner63], looking direction
    fov : list of float, Optional
        Field of view, in angles, from 0-360. Default is [0., 360.].
    method : {"mean", "sum"}, Optional
        Method. Default is "mean".

    Returns
    -------
    out : xarray.DataArray
        HPCA ion flux averaged over the anodes within the selected field of
        view.

    """

    if fov is None:
        fov = [0., 360.]

    assert method in ["mean", "sum"]

    times = inp.time.data
    energies = inp.ccomp.data

    cond_ = np.logical_and(anodes_theta >= fov[0], anodes_theta <= fov[1])
    anodes_in_fov = np.where(cond_)[0]

    if method == "mean":
        updated_spectra = inp.data[:, anodes_in_fov, :].mean(axis=1)
    else:
        updated_spectra = inp.data[:, anodes_in_fov, :].sum(axis=1)

    out = xr.DataArray(updated_spectra, coords=[times, energies],
                       dims=["time", "energy"])

    return out
