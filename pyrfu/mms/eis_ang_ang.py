#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import itertools

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf import ts_skymap

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _check_spin(spin):
    _, spin_inds = np.unique(spin, return_index=True)

    # Check to see if we have a complete first spin,
    # if not then go to the second spin
    if np.where(spin == spin[spin_inds[0]])[0].size != 8:
        spin_inds = spin_inds[1:]

    # Check to see if the last one is complete,
    # if not then go to the second spin
    if np.where(spin == spin[spin_inds[-1]])[0].size != 8:
        spin_inds = spin_inds[:-1]

    return spin_inds


def eis_ang_ang(inp_allt, en_chan: list = None):
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

    azi_ = np.zeros((len(scopes), len(inp_allt.time)))
    pol_ = np.zeros((len(scopes), len(inp_allt.time)))

    for i, scope in enumerate(scopes):
        d_ = inp_allt[f"look_{scope}"]
        # Domain [-180,180], 0 = sunward (GSE)
        azi_[i, :] = np.rad2deg(np.arctan(d_.data[:, 0] / d_.data[:, 1]))
        # Domain [-90,90], Positive is look direction northward
        pol_[i, :] = 90. - np.rad2deg(np.arccos(d_[:, 2]))

    spin_ = inp_allt.spin.data
    sect_ = inp_allt.sector.data

    spin_inds = _check_spin(spin_)

    n_spins, n_pol, n_azi = [len(spin_inds), 6, len(np.unique(sect_))]

    # Minus 80 plus
    min_pol_edges = -80. + 160. * np.arange(n_pol) / n_pol
    max_pol_edges = -80. + 160. * (np.arange(n_pol) + 1) / n_pol

    min_azi_edges = -180. + 360. * np.arange(n_azi) / n_azi
    max_azi_edges = -180. + 360. * (np.arange(n_azi) + 1) / n_azi

    out_data = np.zeros((n_spins, n_en, n_azi, n_pol))

    time_data = time_[spin_inds]

    for i, spin_ind in enumerate(spin_inds):
        t_inds = np.where(spin_ == spin_[spin_ind])[0]
        for t_ind, (t, scope) in itertools.product(t_inds, enumerate(scopes)):
            cond_azi = np.logical_and(azi_[t, t_ind] > min_azi_edges,
                                      azi_[t, t_ind] < max_azi_edges)
            a_idx = np.where(cond_azi)[0]

            cond_pol = np.logical_and(pol_[t, t_ind] > min_pol_edges,
                                      pol_[t, t_ind] < max_pol_edges)
            p_idx = np.where(cond_pol)[0]

            out_data[i, :, a_idx, p_idx] = inp_allt[scope].data[t_ind, en_chan]

    out = xr.DataArray(out_data,
                       coords=[time_data, inp_allt.energy.data[en_chan],
                               min_azi_edges + 180. / n_azi,
                               min_pol_edges + 90. / n_pol],
                       dims=["time", "energy", "phi", "theta"])

    return out
