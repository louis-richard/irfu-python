#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feeps_spin_avg.py

@author : Louis RICHARD
"""

import warnings
import numpy as np
import xarray as xr
from astropy.time import Time

from .db_get_ts import db_get_ts


def feeps_spin_avg(inp_dataset_omni=None):
    """This function will spin-average the omni-directional FEEPS energy spectra.
    
    Parameters
    ----------
    inp_dataset_omni : xarray.DataArray
        Spectrogram of all eyes in OMNI.

    Returns
    -------
    out : xarray.DataArray
        Spin-averaged OMNI energy spectrum.

    """

    if inp_dataset_omni is None:
        raise ValueError("feeps_spin_avg requires at least one argument")

    var = inp_dataset_omni.attrs

    # get the spin sectors
    # v5.5+ = mms1_epd_feeps_srvy_l1b_electron_spinsectnum
    trange = list(Time(inp_dataset_omni.time.data[[0, -1]], format="datetime64").isot)

    dset_name = f"mms{var['mmsId']}_feeps_{var['tmmode']}_{var['lev']}_{var['dtype']}"
    dset_pref = f"mms{var['mmsId']}_epd_feeps_{var['tmmode']}_{var['lev']}_{var['dtype']}"

    spin_sectors = db_get_ts(dset_name, "_".join([dset_pref, "spinsectnum"]), trange)

    spin_starts = [spin_end + 1 for spin_end in np.where(spin_sectors[:-1] >= spin_sectors[1:])[0]]

    energies = inp_dataset_omni.coords["energy"]
    data = inp_dataset_omni.data

    spin_avg_flux = np.zeros([len(spin_starts), len(energies)])

    current_start = spin_starts[0]

    for spin_idx in range(1, len(spin_starts) - 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            spin_avg_flux[spin_idx - 1, :] = np.nanmean(
                data[current_start:spin_starts[spin_idx] + 1, :], axis=0)
        
        current_start = spin_starts[spin_idx] + 1

    time = inp_dataset_omni.coords["time"].data[spin_starts]
    out = xr.DataArray(spin_avg_flux, coords=[time, energies], dims=inp_dataset_omni.dims)

    return out
