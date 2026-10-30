#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .dsl2gse import dsl2gse
from .eis_spin_avg import _spin_segments

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

# Polar bins (colatitudes, degrees): 6 bins from 10 to 170, the elevations
# -80 to 80 of IDL SPEDAS mms_eis_ang_ang
_POL_EDGES = 10.0 + 160.0 * np.arange(7) / 6


def _combine_attrs(inp_allt_attrs):
    attrs = {}
    for k in inp_allt_attrs[0]:
        allt_attrs = [inp_allt_attr[k] for inp_allt_attr in inp_allt_attrs]
        is_same = [allt_attr == allt_attrs[0] for allt_attr in allt_attrs[1:]]

        if all(is_same):
            attrs[k] = allt_attrs[0]
        else:
            continue

    return attrs


def _look_angles(look):
    r"""Azimuth in [0, 360) and polar angle (colatitude) in [0, 180] of the
    look directions, in degrees."""
    look = look / np.linalg.norm(look, axis=-1, keepdims=True)
    phi = np.mod(np.rad2deg(np.arctan2(look[..., 1], look[..., 0])), 360.0)
    theta = np.rad2deg(np.arccos(np.clip(look[..., 2], -1.0, 1.0)))
    return phi, theta


def _angle_attrs(name, frame, valid_max):
    return {
        "CATDESC": f"EPD-EIS sky-map look direction {name} angles",
        "COORDINATE_SYSTEM": frame.upper(),
        "FIELDNAM": f"EPD-EIS {name}",
        "LABLAXIS": {"azimuth": "phi", "polar": "theta"}[name],
        "SI_CONVERSION": "0.0174532925>rad",
        "UNITS": "deg",
        "VALIDMAX": valid_max,
        "VALIDMIN": 0.0,
        "VAR_NOTES": f"Bin centres of the {name} angles of the look directions",
        "VAR_TYPE": "support_data",
    }


def eis_ang_ang(
    inp_allt,
    en_chan: list = None,
    defatt: xr.Dataset = None,
    frame: str = "dbcs",
):
    r"""Generates EIS angle-angle distribution: the fluxes binned in the look
    directions of the telescopes, per spin.

    Parameters
    ----------
    inp_allt : xarray.Dataset
        Dataset of the fluxes of all 6 telescopes, with their look directions
        in GSE (``look_t?``), spin numbers and sectors (from
        :func:`pyrfu.mms.get_eis_allt`).
    en_chan : array_like, Optional
        Energy channels to use. Default use all energy channels.
    defatt : xarray.Dataset, Optional
        Spacecraft attitude (``z_ra``, ``z_dec``), for `frame` "dbcs".
    frame : {"dbcs", "gse"}, Optional
        Frame of the look directions: DBCS (from GSE with `defatt`, as the FPI
        distributions) or GSE (that of the files). Default is "dbcs".

    Returns
    -------
    out : xarray.DataArray
        EIS skymap-like distribution, (time, energy, phi, theta): one per
        complete spin at the time of its first sample, with the azimuth (phi,
        in [0, 360)) and the polar angle (theta) of the look directions.

    Raises
    ------
    ValueError
        If `frame` is not "dbcs" or "gse", if `frame` is "dbcs" and `defatt` is
        not given, or if the interval has no complete spin.

    Notes
    -----
    The angles are those of the look directions of the telescopes (the
    particles move in the opposite directions), as the FPI distributions:
    :func:`pyrfu.mms.psd_moments` and :func:`pyrfu.mms.reduce` take the
    particle velocities opposite to them. The polar angles are binned in 6
    bins from 10 to 170 degrees (the elevations -80 to 80 of IDL SPEDAS
    ``mms_eis_ang_ang``) and the azimuths in as many bins as sectors over
    [0, 360), labelled at their centres. Each cell is the mean of the samples
    of the spin in it, NaN without samples. The spins are those of
    :func:`pyrfu.mms.eis_spin_avg`: complete spins at their first sample.

    The azimuths were in [0, 360] and the polar angles in [0, 180] but binned
    on [-180, 180] and [-80, 80]: about 19 % of the samples were binned, at
    the wrong polar angles (+90 degrees). Empty cells were 0, and the last
    sample in a cell was kept. Without `defatt`, the angles were in GSE.

    """

    if frame.lower() not in ["dbcs", "gse"]:
        raise ValueError(f"frame must be dbcs or gse, not {frame}")

    if frame.lower() == "dbcs" and defatt is None:
        raise ValueError(
            "frame='dbcs' needs the spacecraft attitude (defatt): give it, or use "
            "frame='gse'"
        )

    if en_chan is None:
        en_chan = np.arange(len(inp_allt.energy))

    en_chan = np.atleast_1d(en_chan)
    time_ = inp_allt.time.data

    scopes = list(filter(lambda x: x.startswith("t"), inp_allt.keys()))

    # Look directions (time, telescope, xyz), in DBCS or GSE
    look = []

    for scope in scopes:
        d_xyz = inp_allt[f"look_{scope}"]

        if frame.lower() == "dbcs":
            d_xyz = dsl2gse(d_xyz, defatt, -1)

        look.append(d_xyz.data)

    phi, theta = _look_angles(np.stack(look, axis=1))

    # Bins: as many azimuths as sectors over [0, 360), 6 polar angles
    n_azi = len(np.unique(inp_allt.sector.data))
    azi_edges = 360.0 * np.arange(n_azi + 1) / n_azi
    i_azi = np.clip(np.digitize(phi, azi_edges) - 1, 0, n_azi - 1)
    i_pol = np.digitize(theta, _POL_EDGES) - 1
    in_pol = (i_pol >= 0) & (i_pol < len(_POL_EDGES) - 1)

    # Complete spins, at their first sample
    spin_starts, spin_stops = _spin_segments(inp_allt.spin.data)
    i_spin = np.full(len(time_), -1)

    for i, (i_start, i_stop) in enumerate(zip(spin_starts, spin_stops)):
        i_spin[i_start:i_stop] = i

    # Mean of the samples in each (spin, energy, azimuth, polar) cell
    flux = np.stack([inp_allt[scope].data[:, en_chan] for scope in scopes], axis=1)
    keep = (i_spin[:, None] >= 0) & in_pol
    keep = keep[..., None] & np.isfinite(flux)
    i_t, i_s, i_e = np.nonzero(keep)

    shape = (len(spin_starts), len(en_chan), n_azi, len(_POL_EDGES) - 1)
    sums, counts = np.zeros(shape), np.zeros(shape)
    index = (i_spin[i_t], i_e, i_azi[i_t, i_s], i_pol[i_t, i_s])
    np.add.at(sums, index, flux[i_t, i_s, i_e])
    np.add.at(counts, index, 1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out_data = sums / counts

    # Attributes: those of all the telescopes, the energy widths of the
    # channels used (they were those of all the channels) and the species
    attrs = {
        "species": inp_allt.attrs["species"],
        **{
            k: np.asarray(inp_allt.attrs[k])[en_chan]
            for k in ["delta_energy_plus", "delta_energy_minus"]
        },
        **_combine_attrs([inp_allt[scope].attrs for scope in scopes]),
    }
    attrs = {k: attrs[k] for k in sorted(attrs)}

    out = xr.DataArray(
        out_data,
        coords=[
            time_[spin_starts],
            inp_allt.energy.data[en_chan],
            azi_edges[:-1] + np.diff(azi_edges) / 2,
            _POL_EDGES[:-1] + np.diff(_POL_EDGES) / 2,
        ],
        dims=["time", "energy", "phi", "theta"],
        attrs=attrs,
    )

    # Fill coordinates attributes
    out.time.attrs = inp_allt.time.attrs  # time
    out.energy.attrs = inp_allt.energy.attrs  # energy
    out.phi.attrs = _angle_attrs("azimuth", frame, 360.0)
    out.theta.attrs = _angle_attrs("polar", frame, 180.0)

    return out
