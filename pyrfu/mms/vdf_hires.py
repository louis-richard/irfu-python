#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from pyrfu._core.time import _timedelta
from pyrfu._core.time_attrs import _is_time_attr
from pyrfu.mms._internal.vdf_bins import _check_same_bins, _is_skymap

from ..pyrf.ts_skymap import ts_skymap


def vdf_hires(vdf, offsets, n_parts: int = 4):
    r"""Splits each distribution of a skymap into 2 or 4 parts in time, from
    the time at which each bin was measured.

    Parameters
    ----------
    vdf : xarray.Dataset
        Time series of the skymap distribution, with the half widths of its
        accumulations (``delta_time_minus/plus``, s), e.g. FPI burst.
    offsets : xarray.Dataset
        Time of each bin from the start of its accumulation (us), at the
        times and on the bins of `vdf`, e.g.
        ``get_data("pdtoff{e,i}_fpi_brst_l2", ...)`` (FPI burst files of
        version 3.4 and later).
    n_parts : {2, 4}, Optional
        Number of parts of each accumulation. Default is 4.

    Returns
    -------
    out : xarray.Dataset
        Skymap with `n_parts` distributions per distribution of `vdf`, at the
        centres of the parts: each with the bins measured in its part, NaN
        elsewhere (no data), the energies, azimuths and tables of `vdf`, the
        half width of a part (``delta_time_minus/plus``) and the attribute
        ``n_parts`` (see `hires_moments`).

    Raises
    ------
    TypeError
        If `vdf` and `offsets` are not both xarray.Dataset.
    ValueError
        If `vdf` is not a skymap, `offsets` does not have its times and
        shape, `n_parts` is not 2 or 4, `vdf` has no ``delta_time_minus/plus``,
        or an offset is outside its accumulation.

    Notes
    -----
    The bins of an FPI burst distribution are measured at different times of
    its accumulation (30 ms for DES, 150 ms for DIS). In the burst files
    checked (DES 2019-01-01, DIS 2016-01-11), each quarter holds a quarter of
    the solid angle, spread over all the directions: the azimuths fall into
    four phases, and each quarter holds the equatorial polar bands at one
    phase and the other bands at the others. The highest DES channel is the
    exception, measured in the first and third quarters only (half of the
    solid angle each). A part is a coarse sampling of the directions, not a
    distribution of all the directions at a higher time resolution.

    See Also
    --------
    pyrfu.mms.hires_moments

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2019-01-01T00:18:30.000", "2019-01-01T00:18:31.000"]
    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint, 1)
    >>> off_e = mms.get_data("pdtoffe_fpi_brst_l2", tint, 1)
    >>> vdf_e4 = mms.vdf_hires(vdf_e, off_e, 4)

    """

    if not isinstance(vdf, xr.Dataset) or not isinstance(offsets, xr.Dataset):
        raise TypeError("vdf and offsets must be xarray.Dataset")

    if not _is_skymap(vdf):
        raise ValueError("vdf must be a skymap")

    _check_same_bins(vdf, offsets, "offsets")

    if n_parts not in [2, 4]:
        raise ValueError(f"n_parts must be 2 or 4, not {n_parts!r}")

    if "delta_time_minus" not in vdf.attrs or "delta_time_plus" not in vdf.attrs:
        raise ValueError("vdf must have delta_time_minus and delta_time_plus")

    d_minus = np.asarray(vdf.attrs["delta_time_minus"], dtype=np.float64)
    d_minus = np.broadcast_to(d_minus, (len(vdf.time),))
    d_plus = np.broadcast_to(
        np.asarray(vdf.attrs["delta_time_plus"], dtype=np.float64), (len(vdf.time),)
    )

    # Part of each bin, from its time in the accumulation (us to s)
    duration = (d_minus + d_plus)[:, None, None, None]
    fraction = 1e-6 * np.asarray(offsets.data.data, dtype=np.float64) / duration

    if np.any((fraction < 0) | (fraction >= 1)):
        raise ValueError("the offsets must be within the accumulations")

    part = np.floor(fraction * n_parts).astype(int)
    data = np.asarray(vdf.data.data, dtype=np.float64)

    # Parts of each distribution in turn: (time, part, ...) to (time x part)
    parts = np.stack(
        [np.where(part == k, data, np.nan) for k in range(n_parts)], axis=1
    )
    parts = parts.reshape((-1,) + data.shape[1:])

    # Centres of the parts, from the start of each accumulation
    start = vdf.time.data - _timedelta(d_minus)
    centres = np.outer(d_minus + d_plus, (np.arange(n_parts) + 0.5) / n_parts)
    time = (start[:, None] + _timedelta(centres)).ravel()

    def repeat(values):
        return np.repeat(np.asarray(values), n_parts, axis=0)

    out = ts_skymap(
        time,
        parts,
        repeat(vdf.energy.data),
        repeat(vdf.phi.data),
        vdf.theta.data,
        energy0=vdf.attrs.get("energy0"),
        energy1=vdf.attrs.get("energy1"),
        esteptable=repeat(vdf.attrs["esteptable"]),
    )

    # Attributes: those of each time repeated, the half width of a part
    attrs = {}

    for key, value in vdf.attrs.items():
        if _is_time_attr(key, value, len(vdf.time)):
            attrs[key] = repeat(value)
        else:
            attrs[key] = value

    half = repeat((d_minus + d_plus) / (2 * n_parts))
    attrs.update(
        {
            "delta_time_minus": half,
            "delta_time_plus": half.copy(),
            "esteptable": out.attrs["esteptable"],
            "n_parts": n_parts,
        }
    )
    out.attrs = attrs
    out.time.attrs = dict(vdf.time.attrs)

    for key in vdf:
        out[key].attrs = dict(vdf[key].attrs)

    return out
