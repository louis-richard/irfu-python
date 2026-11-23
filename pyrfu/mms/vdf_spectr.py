#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from pyrfu.mms._internal.vdf_bins import _is_skymap

from ..pyrf.ts_spectr import ts_spectr
from .vdf_omni import vdf_omni

_KINDS = {True: ("energy", "azimuthal"), False: ("pitchangle", "energy")}


def _coordinate(values):
    r"""Coordinate of the spectrogram: one row when it does not change with
    time, else the (time, n) array (a Dataset)."""

    values = np.asarray(values)

    if values.ndim == 2 and np.all(values == values[:1]):
        return values[0]

    return values


def vdf_spectr(vdf, kind: str | None = None, indices=None):
    r"""Spectrogram of a distribution against the energy, the pitch angle or
    the azimuth.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap or pitch-angle distribution (e.g. from `get_pitch_angle_dist`).
    kind : {"energy", "pitchangle", "azimuthal"}, Optional
        Spectrogram against the energy, the pitch angle (pitch-angle
        distribution) or the azimuth (skymap). Default is "pitchangle" for a
        pitch-angle distribution with more than one pitch angle, else
        "energy".
    indices : array_like of int, Optional
        Indices of the bins to average over: the pitch angles for "energy"
        (pitch-angle distribution), the energies for "pitchangle" and
        "azimuthal". Default is all the bins.

    Returns
    -------
    out : xarray.DataArray or xarray.Dataset
        Time series of the spectrogram (time, energy | theta | phi), in the
        units of `vdf`, with its attributes: a DataArray, or a Dataset with
        the coordinate (time, n) when it changes with time (alternating
        energy tables, the azimuths of the FPI burst skymaps), as
        `vdf_omni`; both are taken by `pyrfu.plot.plot_spectr`.

    Raises
    ------
    TypeError
        If `vdf` is not a xarray.Dataset.
    ValueError
        If `kind` is not a spectrogram of `vdf`, or `indices` is given for
        the energy spectrum of a skymap.

    Notes
    -----
    The averages are NaN-means with equal weights: over the pitch angles
    ("energy"), over the energies ("pitchangle"), and over the polar angles
    then the energies ("azimuthal"); the NaN bins have no data and are left
    out, and a time without data is NaN (see :ref:`api-mms-vdf-nan`). The
    energy spectrum of a skymap is `vdf_omni`, weighted by the solid angles;
    for that of a pitch-angle distribution weighted by the solid angles, see
    `pyrfu.pyrf.pitch_angle_average`.

    See Also
    --------
    pyrfu.mms.vdf_omni, pyrfu.pyrf.pitch_angle_average,
    pyrfu.plot.plot_spectr

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint, 1)
    >>> b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint, 1)
    >>> pad_e = mms.get_pitch_angle_dist(vdf_e, b_dmpa, angles=12)

    Pitch-angle spectrogram of the electrons of the channels 10 to 20

    >>> spec = mms.vdf_spectr(pad_e, "pitchangle", indices=range(10, 21))

    """

    if not isinstance(vdf, xr.Dataset):
        raise TypeError("vdf must be a xarray.Dataset")

    skymap = _is_skymap(vdf)
    data = vdf.data.data

    if kind is None:
        kind = "energy" if skymap or data.shape[2] == 1 else "pitchangle"

    if kind not in _KINDS[skymap]:
        raise ValueError(
            f"kind must be one of {', '.join(_KINDS[skymap])} for a "
            f"{'skymap' if skymap else 'pitch-angle distribution'}, not {kind!r}"
        )

    if skymap and kind == "energy":
        if indices is not None:
            raise ValueError("indices are not used for the energy spectrum of a skymap")

        return vdf_omni(vdf)

    # Bins to average over, and the axis of the spectrogram
    if kind == "energy":
        axis, coord, name = 2, vdf.energy.data, "energy"
    elif kind == "pitchangle":
        axis, coord, name = 1, vdf.theta.data, "theta"
    else:
        axis, coord, name = 1, vdf.phi.data, "phi"

    if indices is not None:
        data = np.take(data, np.asarray(indices, dtype=int), axis=axis)

    with warnings.catch_warnings():
        # Times without data: NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if kind == "azimuthal":
            # Polar angles, then energies
            spec = np.nanmean(np.nanmean(data, axis=3), axis=1)
        else:
            spec = np.nanmean(data, axis=axis)

    attrs = {**vdf.data.attrs, **vdf.attrs}
    attrs = {k: attrs[k] for k in sorted(attrs)}
    out = ts_spectr(
        vdf.time.data, _coordinate(coord), spec, comp_name=name, attrs=attrs
    )

    if isinstance(out, xr.Dataset):
        out.data.attrs = dict(vdf.data.attrs)
        out.attrs = dict(vdf.attrs)
    elif name != "energy":
        out[name].attrs["UNITS"] = "deg"

    return out
