#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings
from copy import copy

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

energies_ = {
    "electron": np.array(
        [
            33.2,
            51.90,
            70.6,
            89.4,
            107.1,
            125.2,
            146.5,
            171.3,
            200.2,
            234.0,
            273.4,
            319.4,
            373.2,
            436.0,
            509.2,
        ],
    ),
    "ion": np.array(
        [
            57.9,
            76.8,
            95.4,
            114.1,
            133.0,
            153.7,
            177.6,
            205.1,
            236.7,
            273.2,
            315.4,
            363.8,
            419.7,
            484.2,
            558.6,
        ],
    ),
}


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

    Raises
    ------
    ValueError
        If the integral channel of the eyes is not split (16 channels).

    Notes
    -----
    The integral channel must be split before (`feeps_split_integral_ch`). It
    is better to also remove the bad data and the sunlight contamination, as
    `feeps_corrections` does.

    As in IDL SPEDAS and pyspedas, the channels of an eye whose energy is more
    than 10 % away from the omni-directional energy are not averaged (except
    for SITL data). Eyes with NaN energies are kept, as in IDL and pyspedas.

    See Also
    --------
    pyrfu.mms.get_feeps_alleyes, pyrfu.mms.feeps_remove_bad_data,
    pyrfu.mms.feeps_split_integral_ch, pyrfu.mms.feeps_remove_sun

    """
    d_type, specie = [inp_dataset.attrs["dtype"] for _ in range(2)]
    mms_id = inp_dataset.attrs["mmsId"]
    energies = copy(energies_[d_type])

    # set unique energy bins per spacecraft; from DLT on 31 Jan 2017
    e_corr = {
        "electron": [14.0, -1.0, -3.0, -3.0],
        "ion": [0.0, 0.0, 0.0, 0.0],
    }

    g_fact = {"electron": [1.0, 1.0, 1.0, 1.0], "ion": [0.84, 1.0, 1.0, 1.0]}

    energies += e_corr[specie][mms_id - 1]

    top_sensors = list(filter(lambda x: "top" in x, inp_dataset))
    bot_sensors = list(filter(lambda x: "bot" in x, inp_dataset))

    dalleyes = np.empty(
        (
            len(inp_dataset.time),
            len(energies),
            len(top_sensors) + len(bot_sensors),
        ),
    )
    dalleyes[:] = np.nan

    # percent error around the energy bin centres to accept data for averaging
    en_chk = 0.10

    for idx, sensor in enumerate(top_sensors + bot_sensors):
        if inp_dataset[sensor].shape[1] != len(energies):
            raise ValueError(
                f"{sensor} has {inp_dataset[sensor].shape[1]} energy channels: "
                "split the integral channel first with feeps_split_integral_ch "
                "(or use feeps_corrections)"
            )

        dalleyes[:, :, idx] = inp_dataset[sensor].data

        if inp_dataset.attrs.get("lev") == "sitl":
            continue

        # channels outside energies +/- en_chk * energies are not averaged
        eye_energies = inp_dataset[inp_dataset[sensor].dims[1]].data
        bad_energies = np.abs(energies - eye_energies) > en_chk * energies
        dalleyes[:, bad_energies, idx] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        flux_omni = np.nanmean(dalleyes, axis=2)

    flux_omni *= g_fact[specie][mms_id - 1]

    attrs = {
        "dtype": inp_dataset.attrs["dtype"],
        "UNITS": inp_dataset[top_sensors[0]].attrs["UNITS"],
        "species": inp_dataset.attrs["species"],
        "mmsId": inp_dataset.attrs["mmsId"],
        "units_name": inp_dataset.attrs["units_name"],
    }

    flux_omni = xr.DataArray(
        flux_omni,
        coords=[inp_dataset.time.data, energies],
        dims=["time", "energy"],
        attrs=attrs,
    )

    return flux_omni
