#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf.ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"


def _movmean(inp, n_mov_mean):
    # Moving mean along the first axis as MATLAB movmean (shrinking window at
    # the ends): a window length k (centred, k // 2 back and (k - 1) // 2
    # forward) or [back, forward]
    if np.ndim(n_mov_mean) == 0:
        back, forward = int(n_mov_mean) // 2, (int(n_mov_mean) - 1) // 2
    else:
        back, forward = (int(n) for n in n_mov_mean)

    n_t = inp.shape[0]
    cumsum = np.concatenate([np.zeros((1,) + inp.shape[1:]), np.cumsum(inp, axis=0)])
    start = np.clip(np.arange(n_t) - back, 0, n_t)
    stop = np.clip(np.arange(n_t) + forward + 1, 0, n_t)
    count = (stop - start).reshape((-1,) + (1,) * (inp.ndim - 1))
    return (cumsum[stop] - cumsum[start]) / count


def vdf_find_low_counts(
    vdf_counts, counts: float = 5, n_mov_mean=1, output: str = "energy"
):
    r"""Finds the energy channels with few counts, as irfu-matlab
    PDist.find_low_counts.

    Parameters
    ----------
    vdf_counts : xarray.Dataset
        Distribution of counts (time, energy, ...), e.g. for FPI L2 the
        phase-space density over its error squared, (f / f_err)^2 with
        ``mms.get_data("pderri_fpi_brst_l2", ...)``.
    counts : float, Optional
        Limit on the counts of an energy channel, summed over the angles.
        Default is 5.
    n_mov_mean : int or array_like, Optional
        Moving mean of the summed counts in time, as MATLAB movmean: a window
        length, or [back, forward] numbers of samples. Default is 1 (none).
    output : {"energy", "mat"}, Optional
        "energy" (default): the energy of the first channel with more than
        `counts` counts at each time (NaN if none), to mask the channels
        below it with ``vdf_mask(vdf, below=...)``. "mat": (time, energy) 1
        where the counts are below `counts`, 0 elsewhere, for
        ``vdf_mask(vdf, mask=...)``.

    Returns
    -------
    out : xarray.DataArray
        Energies (time series, eV) or (time, energy) mask.

    Raises
    ------
    ValueError
        If `output` is not "energy" or "mat".

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 1)
    >>> err_i = mms.get_data("pderri_fpi_brst_l2", tint, 1)
    >>> counts_i = vdf_i.copy(deep=True)
    >>> counts_i.data.data = (vdf_i.data.data / err_i.data.data) ** 2
    >>> e_low = mms.vdf_find_low_counts(counts_i, 5, 3)
    >>> vdf_i = mms.vdf_mask(vdf_i, below=e_low)

    """
    if output not in ["energy", "mat"]:
        raise ValueError(f"Invalid output {output}: energy or mat")

    data = vdf_counts.data.data
    summed = np.nansum(data.reshape(data.shape[:2] + (-1,)), axis=2)
    summed = _movmean(summed, n_mov_mean)

    if output == "mat":
        return xr.DataArray(
            (summed < counts).astype(int),
            coords=[vdf_counts.time.data, np.arange(data.shape[1])],
            dims=["time", "idx0"],
        )

    energy = np.broadcast_to(
        np.asarray(vdf_counts.energy.data, dtype=np.float64), data.shape[:2]
    )
    above = summed > counts
    first = np.argmax(above, axis=1)
    out = np.where(np.any(above, axis=1), energy[np.arange(len(first)), first], np.nan)
    return ts_scalar(vdf_counts.time.data, out, attrs={"UNITS": "eV"})
