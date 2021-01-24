#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

import warnings
import numpy as np
import xarray as xr

from astropy.time import Time

from .db_get_ts import db_get_ts


def feeps_spin_avg(inp_dataset_omni):
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

    var = inp_dataset_omni.attrs

    # get the spin sectors
    # v5.5+ = mms1_epd_feeps_srvy_l1b_electron_spinsectnum
    trange = list(Time(inp_dataset_omni.time.data[[0, -1]], format="datetime64").isot)

    dataset_name = f"mms{var['mmsId']}_feeps_{var['tmmode']}_{var['lev']}_{var['dtype']}"
    dataset_pref = f"mms{var['mmsId']}_epd_feeps_{var['tmmode']}_{var['lev']}_{var['dtype']}"

    spin_sectors = db_get_ts(dataset_name, "_".join([dataset_pref, "spinsectnum"]), trange)

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
