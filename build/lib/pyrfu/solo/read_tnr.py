#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
import pandas as pd

from cdflib import cdfepoch, cdfread
from dateutil import parser as date_parser

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _datetime_to_tt2000(time):
    r"""Converts datetime.datetime into epoch_tt_2000.

    Parameters
    ----------
    time : datetime.datetime or list of datetime.datetime
        Time or list of times to convert.

    Returns
    -------
    tt2000 : str
        Time into TT2000 format.

    """

    time_ = pd.Timestamp(time)

    # Convert to string
    return f"{time_.strftime('%Y-%m-%dT%H:%M:%S.%f')}{time_.nanosecond:03d}"


def _rcdf(path, tint):
    r"""Reads CDF files.

    Parameters
    ----------
    path : str
        String of the filename in .cdf containing the L2 data
    tint : list
        Time interval

    Returns
    -------
    out_dict : dict
        Hash table with fields contained in the .cdf file.

    """
    out_dict = {}

    with cdfread.CDF(path) as file:
        keys_ = file.cdf_info()["zVariables"]
        for k_ in keys_:
            temp_ = file.varget(k_, starttime=tint[0], endtime=tint[1])
            shape_ = temp_.shape
            coords = [np.arange(lim_) for lim_ in shape_]

            out_dict[k_.lower()] = xr.DataArray(temp_, coords=coords)

    return out_dict


def read_tnr(path, tint, sensor: int = 4):
    r"""Read L2 data from TNR

    Parameters
    ----------
    path : str
        String of the filename in .cdf containing the L2 data
    tint : list
        Time interval
    sensor : int, Optional
        TNR sensor to be read:
            * 1: V1
            * 2: V2
            * 3: V3,
            * 4: V1 - V2
            * 5: V2 - V3
            * 6: V3 - V1
            * 7: B

    Returns
    -------
    out : xarray.DataArray
        Spectrum of the measured signals.

    Notes
    -----
    The script check if there are data from the two channel and put them
    together.

    """

    tint_ = list(map(date_parser.parse, tint))
    tint_ = list(map(_datetime_to_tt2000, tint_))
    tint_ = list(map(cdfepoch.parse, tint_))

    data_l2 = _rcdf(path, tint_)

    n_freqs = 4 * data_l2["tnr_band_freq"].shape[1]
    freq_tnr = np.reshape(data_l2["tnr_band_freq"].data, n_freqs) / 1000

    epoch_ = data_l2["epoch"].data
    auto1_ = data_l2["auto1"].data
    auto2_ = data_l2["auto2"].data
    sweep_ = data_l2["sweep_num"].data
    bande_ = data_l2["tnr_band"].data
    confg_ = data_l2["sensor_config"].data

    if sensor == 7:
        auto1_ = data_l2["magnetic_spectral_power1"].data
        auto2_ = data_l2["magnetic_spectral_power2"].data

    puntical = np.where(data_l2["front_end"].data == 1)[0]

    epoch_ = epoch_[puntical]
    confg_ = confg_[puntical, :]
    auto1_ = auto1_[puntical, :]
    auto2_ = auto2_[puntical, :]
    sweep_ = sweep_[puntical]
    bande_ = bande_[puntical]

    sweep_num = sweep_

    delta_sw = np.abs(sweep_[1:] - sweep_[:-1])

    xdelta_sw = np.where(delta_sw > 100)[0]
    if xdelta_sw.size:
        xdelta_sw = np.hstack([xdelta_sw, len(sweep_) - 1])
        nxdelta_sw = len(xdelta_sw)
        for inswn in range(nxdelta_sw - 1):
            idx_l, idx_r = [xdelta_sw[inswn] + 1, xdelta_sw[inswn + 1]]
            sweep_num[idx_l:idx_r] += sweep_num[xdelta_sw[inswn]]

    timet_ = cdfepoch.to_datetime(epoch_, to_np=True)

    sens0_, sens1_ = [np.where(confg_[:, i] == sensor)[0] for i in range(2)]

    if sens0_.size and sens1_.size:
        auto_calib = np.vstack([auto1_[sens0_, :], auto2_[sens1_, :]])
        sens_ = np.hstack([sens0_, sens1_])
        timet_ici = np.hstack([timet_[sens0_], timet_[sens1_]])
    elif sens0_.size:
        auto_calib = auto1_[sens0_, :]
        sens_ = sens0_
        timet_ici = timet_[sens0_]
    elif sens1_.size:
        auto_calib = auto2_[sens1_, :]
        sens_ = sens1_
        timet_ici = timet_[sens1_]

    else:
        raise ValueError("no data at all ?!?")

    ord_time = np.argsort(timet_ici)
    time_rr = timet_ici[ord_time]
    sens_ = sens_[ord_time]
    auto_calib = auto_calib[ord_time, :]

    bande_e = bande_[sens_]
    max_sweep = np.max(sweep_num[sens_])
    min_sweep = np.min(sweep_num[sens_])
    sweep_num = sweep_num[sens_]

    v_, time, sweepn_tnr = [[], [], []]

    for ind_sweep in range(min_sweep, max_sweep):
        v1_ = np.zeros(128)
        p_punt = np.where(sweep_num == ind_sweep)[0]
        if p_punt.size:
            for indband in range(p_punt.size):
                idx_l = 32 * bande_e[p_punt[indband]]
                idx_r = 32 * (bande_e[p_punt[indband]] + 1)
                v1_[idx_l:idx_r] = auto_calib[p_punt[indband], :]

            if np.sum(v1_) > 0.0:
                punt0_ = np.where(v1_ == 0.0)[0]
                if punt0_.size:
                    v1_[punt0_] = np.nan
                v_.append(v1_)
                sweepn_tnr.append(sweep_num[p_punt[0]])

        if p_punt.size:
            time.append(time_rr[np.min(p_punt)])

    out = xr.DataArray(np.stack(v_), coords=[np.stack(time), freq_tnr])

    return out
