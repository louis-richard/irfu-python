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

from .get_feeps_active_eyes import get_feeps_active_eyes
from .get_feeps_oneeye import get_feeps_oneeye


# noinspection PyUnboundLocalVariable
def get_feeps_omni(tar_var, tint, mms_id, verbose=True, data_path=""):
    """Computes omni directional energy spectrum of the target data unit for the target specie
    over the target energy range.

    Parameters
    ----------
    tar_var : str
        Key of the target variable like {data_unit}_{specie}_{data_rate}_{data_lvl}.

    tint : list of str
        Time interval.

    mms_id : int or str
        Index of the spacecraft.

    verbose : bool
        Set to True to follow the loading. Default is True.

    data_path : str, optional
        Path of MMS data. Default uses `pyrfu.mms.mms_config.py`

    Returns
    --------
    out : xarray.DataArray
        Energy spectrum of the target data unit for the target specie in omni direction.

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint_brst = ["2017-07-23T16:54:24.000", "2017-07-23T17:00:00.000"]

    Read electron FEEPS omnidirectional energy spectrum

    >>> feeps_omni = mms.get_feeps_omni("flux_ion_brst_l2", tint_brst, 2)

    """

    if isinstance(mms_id, str):
        mms_id = int(mms_id)

    data_units = tar_var.split("_")[0]

    var = {"dtype": tar_var.split("_")[1], "tmmode": tar_var.split("_")[2],
           "lev": tar_var.split("_")[3]}

    specie = var["dtype"][0]

    if var["dtype"] == "electron":
        energies = np.array([33.2, 51.90, 70.6, 89.4, 107.1, 125.2, 146.5, 171.3, 200.2, 234.0,
                             273.4, 319.4, 373.2, 436.0, 509.2, 575.8])
    else:
        energies = np.array([57.9, 76.8, 95.4, 114.1, 133.0, 153.7, 177.6, 205.1, 236.7, 273.2,
                             315.4, 363.8, 419.7, 484.2,  558.6, 609.9])

    # set unique energy bins per spacecraft; from DLT on 31 Jan 2017
    e_corr = {"e": [14.0, -1.0, -3.0, -3.0], "i": [0.0, 0.0, 0.0, 0.0]}

    g_fact = {"e": [1.0, 1.0, 1.0, 1.0], "i": [0.84, 1.0, 1.0, 1.0]}

    energies += e_corr[specie][mms_id - 1]

    active_eyes = get_feeps_active_eyes(var, tint, mms_id)

    # percent error around energy bin center to accept data for averaging; 
    # anything outside of energies[i] +/- en_chk*energies[i] will be changed 
    # to NAN and not averaged   
    # en_chk = 0.1

    top_sensors = active_eyes["top"]
    bot_sensors = active_eyes["bottom"]

    top_it, bot_it = [{}, {}]

    for tsen in top_sensors:
        top = get_feeps_oneeye(f"{data_units}{var['dtype'][0]}_{var['tmmode']}_{var['lev']}",
                               f"top-{tsen:d}", tint, mms_id, verbose, data_path)

        msk = get_feeps_oneeye(f"mask{var['dtype'][0]}_{var['tmmode']}_{var['lev']}",
                               f"top-{tsen:d}", tint, mms_id, verbose, data_path)

        msk.data = np.tile(msk.data[:, 0], (msk.shape[1], 1)).T

        top.data[msk.data == 1] = np.nan

        top_it[tsen] = top

    for bsen in bot_sensors:
        bot = get_feeps_oneeye(f"{data_units}{var['dtype'][0]}_{var['tmmode']}_{var['lev']}",
                               f"bottom-{bsen:d}", tint, mms_id, verbose, data_path)

        msk = get_feeps_oneeye(f"mask{var['dtype'][0]}_{var['tmmode']}_{var['lev']}",
                               f"bottom-{bsen:d}", tint, mms_id, verbose, data_path)

        msk.data = np.tile(msk.data[:, 0], (msk.shape[1], 1)).T

        bot.data[msk.data == 1] = np.nan

        bot_it[bsen] = bot

    dalleyes = np.empty((top_it[top_sensors[0]].shape[0], top_it[top_sensors[0]].shape[1],
                         len(top_sensors) + len(bot_sensors)))

    for i, tsen in enumerate(top_sensors):
        dalleyes[..., i] = top_it[tsen]

    for i, bsen in enumerate(bot_sensors):
        dalleyes[..., len(top_sensors) + i] = bot_it[bsen]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        flux_omni = np.nanmean(dalleyes, axis=2)

    flux_omni *= g_fact[specie][mms_id - 1]

    time = top_it[top_sensors[0]].time.data

    out = xr.DataArray(flux_omni[:], coords=[time, energies], dims=["time", "energy"])

    return out
