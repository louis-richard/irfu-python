#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from copy import copy
from typing import Literal

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from pyrfu.mms._internal.epd import _nan_reduce

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


# set unique energy bins per spacecraft; from DLT on 31 Jan 2017
e_corr_ = {"electron": [14.0, -1.0, -3.0, -3.0], "ion": [0.0, 0.0, 0.0, 0.0]}

# correction factors of the omni-directional data (0.84 for the MMS1 ions)
g_fact_ = {"electron": [1.0, 1.0, 1.0, 1.0], "ion": [0.84, 1.0, 1.0, 1.0]}


def _eyes_on_omni_energies(inp_dataset):
    r"""Puts the eyes on the omni-directional energies.

    Parameters
    ----------
    inp_dataset : xarray.Dataset
        Dataset with all active telescopes data.

    Returns
    -------
    eyes : list of str
        Names of the eyes, top then bottom.
    dalleyes : numpy.ndarray
        Data of the eyes (time, energy, eye), NaN where the eye energy is more
        than 10 % away from the omni-directional energy (not for SITL data).
    energies : numpy.ndarray
        Omni-directional energies of the spacecraft.
    g_fact : float
        Correction factor of the spacecraft, applied to the omni-directional
        data.

    Raises
    ------
    ValueError
        If the integral channel of the eyes is not split (16 channels).

    """

    d_type = inp_dataset.attrs["dtype"]
    mms_id = inp_dataset.attrs["mmsId"]
    energies = copy(energies_[d_type])
    energies += e_corr_[d_type][mms_id - 1]

    top_sensors = list(filter(lambda x: "top" in x, inp_dataset))
    bot_sensors = list(filter(lambda x: "bot" in x, inp_dataset))
    eyes = top_sensors + bot_sensors

    dalleyes = np.empty((len(inp_dataset.time), len(energies), len(eyes)))
    dalleyes[:] = np.nan

    # Percent error around the energy bin centres to accept data for averaging
    en_chk = 0.10

    for idx, sensor in enumerate(eyes):
        if inp_dataset[sensor].shape[1] != len(energies):
            raise ValueError(
                f"{sensor} has {inp_dataset[sensor].shape[1]} energy channels: "
                "split the integral channel first with feeps_split_integral_ch "
                "(or use feeps_corrections)"
            )

        dalleyes[:, :, idx] = inp_dataset[sensor].data

        if inp_dataset.attrs.get("lev") == "sitl":
            continue

        # Channels outside energies +/- en_chk * energies are not averaged
        eye_energies = inp_dataset[inp_dataset[sensor].dims[1]].data
        bad_energies = np.abs(energies - eye_energies) > en_chk * energies
        dalleyes[:, bad_energies, idx] = np.nan

    return eyes, dalleyes, energies, g_fact_[d_type][mms_id - 1]


# Combination of the eyes by default for the units of get_feeps_alleyes:
# the counts are summed, the intensity errors added in quadrature
_DEFAULT_METHODS = {"counts": "sum", "fluxerr": "quadrature"}


def feeps_omni(
    inp_dataset, *, method: Literal["mean", "sum", "quadrature"] | None = None
):
    r"""Calculates the omni-directional FEEPS spectrogram.

    Parameters
    ----------
    inp_dataset : xarray.Dataset
        Dataset with all active telescopes data.
    method : {"mean", "sum", "quadrature"}, Optional
        Combination of the eyes at each time and energy: the mean (intensities,
        count rates), the sum (counts) or the square root of the sum of the
        squares divided by the number of eyes (errors of the mean). Default is
        from the units of `inp_dataset` (``units_name``): "sum" for "counts",
        "quadrature" for "fluxerr", "mean" otherwise.

    Returns
    -------
    flux_omni : xarray.DataArray
        Omni-directional FEEPS spectrogram.

    Raises
    ------
    ValueError
        If the integral channel of the eyes is not split (16 channels), or if
        `method` is not one of the above.

    See Also
    --------
    pyrfu.mms.get_feeps_alleyes, pyrfu.mms.feeps_remove_bad_data,
    pyrfu.mms.feeps_split_integral_ch, pyrfu.mms.feeps_remove_sun

    Notes
    -----
    The integral channel must be split before (`feeps_split_integral_ch`). It
    is better to also remove the bad data and the sunlight contamination, as
    `feeps_corrections` does.

    As in IDL SPEDAS and pyspedas, the channels of an eye whose energy is more
    than 10 % away from the omni-directional energy are not averaged (except
    for SITL data). Eyes with NaN energies are kept, as in IDL and pyspedas.

    The eyes without data (NaN) are left out of the combination, which is NaN
    without any eye. The correction factor of the spacecraft (0.84 for the
    MMS1 ions) multiplies the result whatever the method, as IDL SPEDAS does
    whatever the units: the one-count level (omni intensity / omni counts) is
    unchanged. The errors of the mean ("quadrature") assume independent eyes;
    IDL SPEDAS and pyspedas average every unit, also the counts.

    """

    combine = method or _DEFAULT_METHODS.get(
        inp_dataset.attrs.get("units_name"), "mean"
    )

    if combine not in ["mean", "sum", "quadrature"]:
        raise ValueError(f"method must be mean, sum or quadrature, not {combine!r}")

    eyes, dalleyes, energies, g_fact = _eyes_on_omni_energies(inp_dataset)

    omni_data = _nan_reduce(dalleyes, 2, combine) * g_fact

    attrs = {
        "dtype": inp_dataset.attrs["dtype"],
        "species": inp_dataset.attrs["species"],
        "mmsId": inp_dataset.attrs["mmsId"],
        "units_name": inp_dataset.attrs["units_name"],
    }

    flux_omni = xr.DataArray(
        omni_data,
        coords=[inp_dataset.time.data, energies],
        dims=["time", "energy"],
        attrs=attrs,
    )

    # Units of the eyes, if any; energies in keV
    if "UNITS" in inp_dataset[eyes[0]].attrs:
        flux_omni.attrs["UNITS"] = inp_dataset[eyes[0]].attrs["UNITS"]

    flux_omni.energy.attrs["UNITS"] = "keV"

    return flux_omni
