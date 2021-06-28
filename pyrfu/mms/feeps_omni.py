#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"

energies_ = {"electron": np.array([33.2, 51.90, 70.6, 89.4, 107.1, 125.2,
                                   146.5, 171.3, 200.2, 234.0, 273.4, 319.4,
                                   373.2, 436.0, 509.2]),
             "ion": np.array([57.9, 76.8, 95.4, 114.1, 133.0, 153.7, 177.6,
                              205.1, 236.7, 273.2, 315.4, 363.8, 419.7,
                              484.2, 558.6])}


def feeps_omni(inp_dataset):
    r"""Calculates the omni-directional FEEPS spectrogram.

    Parameters
    ----------
    inp_dataset : xarray.Dataset
        Dataset with all active telescopes data.

    Returns
    -------
    flux_omni : xarray.DataArray
        Omni-directional FEEPS spectrogram.

    Notes
    -----
    The dataset can be raw data, but it is better to remove bad datas,
    sunlight contamination and split before.

    See Also
    --------
    pyrfu.mms.get_feeps_alleyes, pyrfu.mms.feeps_remove_bad_data,
    pyrfu.mms.feeps_split_integral_ch, pyrfu.mms.feeps_remove_sun

    """
    d_type, specie = [inp_dataset.attrs["dtype"] for _ in range(2)]
    mms_id = inp_dataset.attrs["mmsId"]
    energies = energies_[d_type]

    # set unique energy bins per spacecraft; from DLT on 31 Jan 2017
    e_corr = {"electron": [14.0, -1.0, -3.0, -3.0],
              "ion": [0.0, 0.0, 0.0, 0.0]}

    g_fact = {"electron": [1.0, 1.0, 1.0, 1.0], "ion": [0.84, 1.0, 1.0, 1.0]}

    energies += e_corr[specie][mms_id - 1]

    top_sensors = list(filter(lambda x: "top" in x, inp_dataset))
    bot_sensors = list(filter(lambda x: "bot" in x, inp_dataset))

    dalleyes = np.empty((len(inp_dataset.time),
                         len(energies),
                         len(top_sensors) + len(bot_sensors)))
    dalleyes[:] = np.nan

    for idx, sensor in enumerate(top_sensors):
        data = inp_dataset[sensor].data
        dalleyes[:, :, idx] = data

    for idx, sensor in enumerate(bot_sensors):
        data = inp_dataset[sensor].data
        dalleyes[:, :, idx + len(top_sensors)] = data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        flux_omni = np.nanmean(dalleyes, axis=2)

    flux_omni *= g_fact[specie][mms_id - 1]

    flux_omni = xr.DataArray(flux_omni,
                             coords=[inp_dataset.time.data, energies],
                             dims=["time", "energy"])

    return flux_omni
