#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings

# Built-in imports
from typing import Literal

# 3rd party imports
import numpy as np
import xarray as xr

anodes_theta = np.array(
    [
        123.75000,
        101.25000,
        78.750000,
        56.250000,
        33.750000,
        11.250000,
        191.25000,
        213.75000,
        236.25000,
        258.75000,
        281.25000,
        303.75000,
        326.25000,
        348.75000,
        168.75000,
        146.25000,
    ],
)


def hpca_calc_anodes(
    inp, fov: list | None = None, method: Literal["mean", "sum"] = "mean"
):
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
        view. The anodes without data (NaN) are left out; NaN where no anode
        has data.

    Raises
    ------
    ValueError
        If method is not "mean" or "sum".

    """

    if fov is None:
        fov = [0.0, 360.0]

    if method not in ["mean", "sum"]:
        raise ValueError(f'method must be "mean" or "sum", not {method!r}')

    times = inp.time.data
    energies = inp.ccomp.data

    cond_ = np.logical_and(anodes_theta >= fov[0], anodes_theta <= fov[1])
    anodes_in_fov = np.where(cond_)[0]

    spectra = inp.data[:, anodes_in_fov, :]
    has_data = np.isfinite(spectra).any(axis=1)

    with warnings.catch_warnings():
        # NaN without warning where no anode has data
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if method == "mean":
            updated_spectra = np.nanmean(spectra, axis=1)
        else:
            updated_spectra = np.where(has_data, np.nansum(spectra, axis=1), np.nan)

    # The attributes of inp and of its energies (lost)
    out = xr.DataArray(
        updated_spectra,
        coords=[times, energies],
        dims=["time", "energy"],
        attrs=dict(inp.attrs),
    )
    out.energy.attrs = dict(inp.ccomp.attrs)

    return out
