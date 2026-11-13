#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from pyrfu._core.time import _ns
from pyrfu.mms._internal.epd import _spin_reduce

# Start azimuths of HPCA per spin (half spin of the spacecraft), as hpca_pad
_N_AZ = 16


def _hpca_complete_spins(start_az, time):
    r"""Start and stop indices of the complete spins: from one start (start
    azimuth index 0) to the next, with one sample per start azimuth (0, 1,
    ..., 15, in order) and no data gap (two samples more than 1.5 median
    sampling periods apart)."""
    spin_starts = np.flatnonzero(start_az == 0)

    if len(spin_starts) < 2:
        raise ValueError(
            "saz must have at least two spin starts (start azimuth index 0): "
            "no complete spin"
        )

    d_t = np.diff(_ns(time))
    no_gap = d_t <= 1.5 * np.median(d_t)

    starts, stops = spin_starts[:-1], spin_starts[1:]
    complete = [
        stop - start == _N_AZ
        and np.array_equal(start_az[start:stop], np.arange(_N_AZ))
        and np.all(no_gap[start : stop - 1])
        for start, stop in zip(starts, stops)
    ]

    if not np.any(complete):
        raise ValueError("no complete spin (one sample per start azimuth)")

    return starts[complete], stops[complete]


def hpca_spin_sum(inp, saz, method: str = "mean"):
    r"""Sum or average the Hot Plasma Composition Analyser (HPCA) data over
    each spin.

    Parameters
    ----------
    inp : xarray.DataArray
        Ion PSD or flux; [nt, npo16, ner63], looking direction
    saz : xarray.DataArray
        Start azimuthal spin indices, at the times of `inp`.
    method : {"mean", "sum"}, Optional
        Method either "sum" or "mean". Default is "mean"

    Returns
    -------
    out : xarray.DataArray
        Distribution averaged (or summed) over each complete spin, at the time
        of its start.

    Raises
    ------
    ValueError
        If method is not "mean" or "sum", if saz is not at the times of inp,
        or if saz has fewer than two spin starts or no complete spin.

    Notes
    -----
    A spin runs from one start (start azimuth index 0) to the next: the
    samples after the last start, a spin not known to be complete, are not
    used, as pyspedas ``mms_hpca_spin_sum`` (IDL SPEDAS gives a spin of zeros
    there). Only the complete spins are used, with one sample per start
    azimuth (0, 1, ..., 15, in order) and no data gap (two samples more than 1.5
    median sampling periods apart): the spins cut by data gaps were kept as
    partial sums or means, as the EIS and FEEPS spin averages do not. saz is
    used by position, so it must be at the times of inp (misaligned spins
    were used). The samples without data (NaN) are left out; a spin without
    data is NaN.

    """

    if method not in ("mean", "sum"):
        raise ValueError(f"Invalid method {method!r}, use mean or sum")

    if len(saz) != len(inp) or not np.array_equal(saz.time.data, inp.time.data):
        raise ValueError("saz must be at the times of inp")

    starts, stops = _hpca_complete_spins(np.asarray(saz.data), inp.time.data)

    # NaN for a spin without data (the sum was 0, as data)
    out_data = _spin_reduce(inp.data, starts, stops, method)
    out_time = inp.time.data[starts]
    coords = [inp.coords[k].data for k in inp.dims[1:]]
    coords = [out_time, *coords]
    dims = list(inp.dims)

    out = xr.DataArray(out_data, coords=coords, dims=dims, attrs=inp.attrs)

    return out
