#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import itertools

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .dsl2gse import dsl2gse

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _check_spin(spin):
    _, spin_inds = np.unique(spin, return_index=True)

    len_spin = np.max(np.diff(spin_inds))

    # Check to see if we have a complete first spin,
    # if not then go to the second spin
    if np.where(spin == spin[spin_inds[0]])[0].size != len_spin:
        spin_inds = spin_inds[1:]

    # Check to see if the last one is complete,
    # if not then go to the second spin
    if np.where(spin == spin[spin_inds[-1]])[0].size != len_spin:
        spin_inds = spin_inds[:-1]

    return spin_inds, len_spin


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


def eis_ang_ang(inp_allt, en_chan: list = None, defatt: xr.Dataset = None):
    r"""Generates EIS angle-angle distribution.

    Parameters
    ----------
    inp_allt : xarray.Dataset
        Dataset of the fluxes of all 6 telescopes.
    en_chan : array_like, Optional
        Energy channels to use. Default use all energy channels.

    Returns
    -------
    out : xarray.DataArray
        EIS skymap-like distribution.

    """

    if en_chan is None:
        en_chan = np.arange(len(inp_allt.energy))

    time_ = inp_allt.time.data

    n_en = len(en_chan)

    scopes = list(filter(lambda x: x.startswith("t"), inp_allt.keys()))

    phi, theta = [np.zeros((len(scopes), len(inp_allt.time))) for _ in range(2)]

    for i, scope in enumerate(scopes):
        d_xyz = inp_allt[f"look_{scope}"]

        # If defatt is given get angles in spacecraft coordinates system
        if defatt is not None:
            d_xyz = dsl2gse(d_xyz, defatt, -1)
            coordinates_system = "DBCS>Despun Body Coordinate System"
        else:
            coordinates_system = "GSE>Geocentric Solar Magnetospheric"

        # Domain [-180, 180], 0 = sunward (GSE)
        # phi[i, :] = (np.rad2deg(np.arctan2(d_xyz.data[:, 1], d_xyz.data[:, 0])))
        # Domain [0, 360], 0 = sunward (GSE)
        phi[i, :] = np.rad2deg(np.arctan2(d_xyz.data[:, 1], d_xyz.data[:, 0])) + 180.0
        # Domain [-90, 90], Positive is look direction northward
        # theta[i, :] = 90.0 - np.rad2deg(np.arccos(d_xyz[:, 2]))
        # Domain [0, 180], Positive is look direction northward
        theta[i, :] = np.rad2deg(np.arccos(d_xyz[:, 2]))

    spin_ = inp_allt.spin.data
    sect_ = inp_allt.sector.data

    spin_inds, len_spin = _check_spin(spin_)

    n_spins, n_pol, n_azi = [len(spin_inds), 6, len(np.unique(sect_))]

    # Minus 80 plus
    min_pol_edges = -80.0 + 160.0 * np.arange(n_pol) / n_pol
    max_pol_edges = -80.0 + 160.0 * (np.arange(n_pol) + 1) / n_pol
    mid_pol_edges = min_pol_edges + 90.0 / n_pol

    min_azi_edges = -180.0 + 360.0 * np.arange(n_azi) / n_azi
    max_azi_edges = -180.0 + 360.0 * (np.arange(n_azi) + 1) / n_azi
    mid_azi_edges = min_azi_edges + 180.0 / n_azi

    out_data = np.zeros((n_spins, n_en, n_azi, n_pol))

    time_data = time_[spin_inds + len_spin // 2]

    for i, spin_ind in enumerate(spin_inds):
        t_inds = np.where(spin_ == spin_[spin_ind])[0]
        for t_ind, (i_s, scope) in itertools.product(
            t_inds,
            enumerate(scopes),
        ):
            cond_azi = np.logical_and(
                phi[i_s, t_ind] > min_azi_edges,
                phi[i_s, t_ind] < max_azi_edges,
            )
            a_idx = np.where(cond_azi)[0]

            cond_pol = np.logical_and(
                theta[i_s, t_ind] > min_pol_edges,
                theta[i_s, t_ind] < max_pol_edges,
            )
            p_idx = np.where(cond_pol)[0]

            out_data[i, :, a_idx, p_idx] = inp_allt[scope].data[t_ind, en_chan]

    # Setup attributes (that of all telescopes + delta energies and
    # particle species)
    attrs = {k: inp_allt.attrs[k] for k in ["delta_energy_plus", "delta_energy_minus"]}
    attrs = {"species": inp_allt.attrs["species"], **attrs}
    attrs = {
        **attrs,
        **_combine_attrs([inp_allt[scope].attrs for scope in scopes]),
    }
    attrs = {k: attrs[k] for k in sorted(attrs)}

    out = xr.DataArray(
        out_data,
        coords=[
            time_data,
            inp_allt.energy.data[en_chan],
            mid_azi_edges + 180.0,
            mid_pol_edges + 90.0,
        ],
        dims=["time", "energy", "phi", "theta"],
        attrs=attrs,
    )

    # Fill coordinates attributes
    out.time.attrs = inp_allt.time.attrs  # time
    out.energy.attrs = inp_allt.energy.attrs  # energy

    # Fill azimuthal angles attributes
    out.phi.attrs = {
        "CATDESC": f"{attrs['CATDESC'][0][:4]} EPD-EIS sky-map azimuth angles",
        "COORDINATE_SYSTEM": coordinates_system,
        "DELTA_MINUS_VAR": " ",
        "DELTA_PLUS_VAR": " ",
        "FIELDNAM": f"{attrs['CATDESC'][0][:4]} EPD-EIS phi",
        "FILLVAL": -1e31,
        "FORMAT": "E12.2",
        "LABLAXIS": "phi",
        "REPRESENTATION_1": "t",
        "SCALETYP": "linear",
        "SI_CONVERSION": "0.0174532925>rad",
        "UNITS": "deg",
        "VALIDMAX": 360.0,
        "VALIDMIN": 0.0,
        "VAR_NOTES": "Azimuth angles in the instrument frame",
        "VAR_TYPE": "support_data",
    }

    # Fill elevation angles attributes
    out.theta.attrs = {
        "CATDESC": f"{attrs['CATDESC'][0][:4]} EPD-EIS sky-map zenith angles",
        "COORDINATE_SYSTEM": coordinates_system,
        "DELTA_MINUS_VAR": " ",
        "DELTA_PLUS_VAR": " ",
        "FIELDNAM": f"{attrs['CATDESC'][0][:4]} EPD-EIS theta",
        "FILLVAL": -1e31,
        "FORMAT": "E12.2",
        "LABLAXIS": "theta",
        "REPRESENTATION_1": "t",
        "SCALETYP": "linear",
        "SI_CONVERSION": "0.0174532925>rad",
        "UNITS": "deg",
        "VALIDMAX": 180.0,
        "VALIDMIN": 0.0,
        "VAR_NOTES": "Pixel zenith angles",
        "VAR_TYPE": "support_data",
    }

    return out
