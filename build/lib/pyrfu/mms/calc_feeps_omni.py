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

"""calc_feeps_omni.py
@author: Louis Richard
"""

import os
import warnings
import yaml
import numpy as np
import xarray as xr

from .feeps_remove_sun import feeps_remove_sun
from .feeps_split_integral_ch import feeps_split_integral_ch


def calc_feeps_omni(inp_dataset):
    """Computes the omni-directional FEEPS spectrogram from a Dataset that
    contains the spectrogram of all eyes.

    Parameters
    ----------
    inp_dataset : xarray.Dataset
        Dataset with energy spectrum of every eyes.

    Returns
    -------
    out : xarray.DataArray
        OMNI energy spectrum from the input.

    """

    var = inp_dataset.attrs

    if var["dtype"] == "electron":
        energies = np.array(
            [33.2, 51.90, 70.6, 89.4, 107.1, 125.2, 146.5, 171.3, 200.2,
             234.0, 273.4, 319.4, 373.2, 436.0, 509.2])
    else:
        energies = np.array(
            [57.9, 76.8, 95.4, 114.1, 133.0, 153.7, 177.6, 205.1, 236.7,
             273.2, 315.4, 363.8, 419.7, 484.2, 558.6])

    # set unique energy bins per spacecraft; from DLT on 31 Jan 2017
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "feeps_calibration.yml")) as file:
        calibration = yaml.load(file, Loader=yaml.FullLoader)

    energies += calibration["e_corr"][var["dtype"][0]][var["mmsId"]-1]

    # top_sensors = eyes['top']
    # bot_sensors = eyes['bottom']

    inp_dataset_clean, _ = feeps_split_integral_ch(inp_dataset)

    inp_dset_csr = feeps_remove_sun(inp_dataset_clean)

    eye_list = list(inp_dset_csr.keys())

    d_alleyes = np.empty((inp_dset_csr[eye_list[0]].shape[0],
                          inp_dset_csr[eye_list[0]].shape[1],
                          len(inp_dset_csr)))
    d_alleyes[:] = np.nan

    for i, k in enumerate(eye_list):
        d_alleyes[..., i] = inp_dset_csr[k].data

        try:
            # percent error around energy bin center to accept data for
            # averaging; anything outside of energies[i] +/- en_chk *
            # energies[i] will be changed to NAN and not averaged

            energy_indices = np.where(
                np.abs(energies - inp_dset_csr[k].coords[
                    "Differential_energy_channels"].data) > .1 * energies)

            if energy_indices[0].size != 0:
                d_alleyes[:, energy_indices[0], i] = np.nan

        except Warning:
            print('NaN in energy table encountered; sensor T{}'.format(k))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        flux_omni = np.nanmean(d_alleyes, axis=2)

    flux_omni *= calibration["g_fact"][var["dtype"][0]][var["mmsId"]-1]

    out = xr.DataArray(flux_omni, coords=[inp_dset_csr.time.data, energies],
                       dims=["time", "energy"],
                       attrs=inp_dset_csr[eye_list[0]].attrs)

    return out
