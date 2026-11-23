#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Literal

# 3rd party imports
import numpy as np
from scipy import interpolate

# Local imports
from pyrfu.mms._internal.vdf_bins import _is_skymap


def _interpolant(energy, values, method):
    r"""Interpolant of values (energy, ...) along the energies, NaN outside
    their range."""

    if method == "pchip":
        return interpolate.PchipInterpolator(energy, values, axis=0, extrapolate=False)

    return interpolate.interp1d(
        energy, values, axis=0, bounds_error=False, fill_value=np.nan
    )


def _interp_time(energy, data, new_energy, method):
    r"""Distribution of one time (energy, phi, theta) on new_energy, each
    bin through the energies where it has data."""

    finite = np.isfinite(data)
    complete = np.all(finite, axis=0)
    out = np.full((len(new_energy),) + data.shape[1:], np.nan)

    out[:, complete] = _interpolant(energy, data[:, complete], method)(new_energy)

    for i_p, i_q in zip(*np.where(~complete)):
        keep = finite[:, i_p, i_q]

        if np.sum(keep) > 1:
            out[:, i_p, i_q] = _interpolant(energy[keep], data[keep, i_p, i_q], method)(
                new_energy
            )

    return out


def vdf_einterp(vdf, method: Literal["pchip", "linear"] = "pchip"):
    r"""Interpolates a skymap with alternating energy tables onto the union of
    its tables at every time, keeping its time resolution.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution (e.g. FPI burst, two energy tables of 32
        channels used in turn).
    method : {"pchip", "linear"}, Optional
        Interpolation in energy of each angular bin: "pchip" (default), the
        shape-preserving piecewise cubic Hermite interpolation, or "linear".

    Returns
    -------
    out : xarray.Dataset
        Skymap on the energies of all the tables (64 for FPI), the same at
        every time (``energy0`` = ``energy1``, ``esteptable`` 0), with the
        energy widths of the new channels (``delta_energy_minus/plus``: to
        the geometric means with the neighbouring energies, the outer ones
        mirrored) and the other attributes of `vdf`.

    Raises
    ------
    TypeError
        If `vdf` is not a xarray.Dataset.
    ValueError
        If `vdf` is not a skymap, has energies not above 0, or `method` is
        not "pchip" or "linear".

    Notes
    -----
    The distribution is interpolated in energy, linearly in its value, and
    negative values are set to 0. At each time, the energies of the union
    outside the range of the table of that time are NaN (no data, see
    :ref:`api-mms-vdf-nan`); a bin with NaN is interpolated through its
    energies with data. The energy resolution doubles but the interpolated
    values are not measurements: their counting errors are those of the
    neighbouring channels, and features narrower than a channel are not
    resolved. `vdf_to_e64` combines two successive distributions instead,
    at half the time resolution, with measured values only.

    See Also
    --------
    pyrfu.mms.vdf_to_e64

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2015-10-30T05:15:45.000", "2015-10-30T05:15:46.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 3)
    >>> vdf_i64 = mms.vdf_einterp(vdf_i)

    """

    if not hasattr(vdf, "data_vars"):
        raise TypeError("vdf must be a xarray.Dataset")

    if not _is_skymap(vdf):
        raise ValueError("vdf must be a skymap")

    if method not in ["pchip", "linear"]:
        raise ValueError(f'method must be "pchip" or "linear", not {method!r}')

    energy = np.asarray(vdf.energy.data, dtype=np.float64)
    new_energy = np.unique(energy)

    if not np.all(new_energy > 0):
        raise ValueError("the energies of vdf must be positive")

    data = np.asarray(vdf.data.data, dtype=np.float64)

    new_data = np.stack(
        [
            _interp_time(energy[i_t], data[i_t], new_energy, method)
            for i_t in range(len(vdf.time))
        ]
    )
    new_data[new_data < 0] = 0.0

    # Widths to the geometric means with the neighbouring energies
    edges = np.sqrt(new_energy[:-1] * new_energy[1:])
    edges = np.hstack(
        [new_energy[0] ** 2 / edges[0], edges, new_energy[-1] ** 2 / edges[-1]]
    )
    n_t = len(vdf.time)

    out = vdf.drop_vars(["data", "energy", "idx0"]).assign(
        data=(vdf.data.dims, new_data, dict(vdf.data.attrs)),
        energy=(
            vdf.energy.dims,
            np.tile(new_energy, (n_t, 1)),
            dict(vdf.energy.attrs),
        ),
    )
    out = out.assign_coords(idx0=np.arange(len(new_energy)))
    out.attrs = {
        **vdf.attrs,
        "energy0": new_energy,
        "energy1": new_energy,
        "esteptable": np.zeros(n_t, dtype=np.uint8),
        "delta_energy_minus": np.tile(new_energy - edges[:-1], (n_t, 1)),
        "delta_energy_plus": np.tile(edges[1:] - new_energy, (n_t, 1)),
    }

    return out
