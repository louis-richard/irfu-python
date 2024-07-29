#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import datetime
import json
import logging
import os
import re
from typing import Optional

# 3rd party imports
import numpy as np
import pycdfpp
import xarray as xr
from dateutil import parser
from dateutil.rrule import DAILY, rrule
from scipy import integrate
from xarray.core.dataarray import DataArray

from ..pyrf import read_cdf, time_clip, ts_append

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


def _list_files_tnr_l2(
    tint: list, data_path: Optional[str] = "", tree: Optional[bool] = False
) -> list:
    """Find files in the L2 data repo corresponding to the target time
    interval.

    Parameters
    ----------
    tint : list
        Time interval
    data_path : str, Optional
        Path of MMS data. Default uses `pyrfu.solo.config.json`
    tree : bool, Optional
        Flag for tree structured data repos. Default is False.

    Returns
    -------
    list
        List of files corresponding to the parameters in the selected time
        interval

    """

    # Check path
    if not data_path:
        pkg_path = os.path.dirname(os.path.abspath(__file__))

        # Read the current version of the MMS configuration file
        with open(os.path.join(pkg_path, "config.json"), "r", encoding="utf-8") as fs:
            config = json.load(fs)

        data_path = os.path.normpath(config["local_data_dir"])
    else:
        data_path = os.path.normpath(data_path)

    # Make sure that the data path exists
    assert os.path.exists(data_path), f"{data_path} doesn't exist!!"

    files_out = []

    # directory and file name search patterns:
    # - assume directories are of the form: [data_path]/L2/thr/year/month/
    # - assume file names are of the form:
    #   solo_L2_rpw-tnr-surv-cdag_YYYYMMDD_version.cdf

    file_name = r"solo_L2_rpw-tnr-surv.*_([0-9]{8})_V[0-9]{2}.cdf"

    # Check tint
    assert isinstance(tint, (list, np.ndarray)), "tint must be array_like"
    assert len(tint) == 2, "tint must contain two elements"
    assert isinstance(tint[0], str), "tint[0] must be a string"
    assert isinstance(tint[1], str), "tint[1] must be a string"

    d_start = parser.parse(parser.parse(tint[0]).strftime("%Y-%m-%d"))
    until_ = parser.parse(tint[1]) - datetime.timedelta(seconds=1)
    days = rrule(DAILY, dtstart=d_start, until=until_)

    for date in days:
        if tree:
            local_dir = os.sep.join(
                [
                    data_path,
                    "L2",
                    "thr",
                    date.strftime("%Y"),
                    date.strftime("%m"),
                ],
            )
        else:
            local_dir = data_path

        if os.name == "nt":
            full_path = os.sep.join(
                [re.escape(local_dir) + os.sep, file_name],
            )
        else:
            full_path = os.sep.join([re.escape(local_dir), file_name])

        regex = re.compile(full_path)

        for root, _, files in os.walk(local_dir):
            for file in files:
                this_file = os.sep.join([root, file])

                matches = regex.match(this_file)
                if matches:
                    this_time = parser.parse(matches.groups()[0])
                    if d_start <= this_time <= until_:
                        if this_file not in files_out:
                            files_out.append(os.sep.join([local_dir, file]))

    # sort in time
    if len(files_out) > 1:
        files_out = sorted(files_out)

    return files_out


def read_tnr(
    tint: list,
    sensor: Optional[int] = 4,
    data_path: Optional[str] = "",
    tree: Optional[bool] = False,
) -> DataArray:
    r"""Read L2 data from TNR

    Parameters
    ----------
    tint : list
        Time interval
    sensor : int, Optional
        TNR sensor to be read:
            * 1: V1
            * 2: V2
            * 3: V3
            * 4: V1 - V2 (default)
            * 5: V2 - V3
            * 6: V3 - V1
            * 7: B
    data_path : str, Optional
        Path of MMS data. Default uses `pyrfu.solo.config.json`
    tree : bool, Optional
        Flag for tree structured data repos. Default is False.

    Returns
    -------
    DataArray
        Spectrum of the measured signals.

    Raises
    ------
    ValueError
        If there is no data from the sensor selected.

    Notes
    -----
    The script check if there are data from the two channel and put them
    together.

    """

    # Check input types
    assert isinstance(sensor, int), "sensor must integer"

    files = _list_files_tnr_l2(tint, data_path, tree)

    # Initialize spectrum output to None
    out = None

    for file in files:
        # Notify user
        logging.info("Loading %s...", os.path.split(file)[-1])

        data_l2 = read_cdf(file)

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

        # Convert epoch to datetime64
        timet_ = pycdfpp.to_datetime64(epoch_)

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

        time = time[1:]
        v_ = np.stack(v_)

        # select frequencies lower than 100 kHz
        freq_tnr = freq_tnr[freq_tnr < 1e2]
        vp = v_[1:, : len(freq_tnr)]

        # Integration
        itg = integrate.trapz(vp, axis=1) / vp.shape[0]
        vp = vp - itg[:, None]

        out = ts_append(
            out,
            xr.DataArray(
                vp,
                coords=[np.stack(time), freq_tnr],
                dims=["time", "frequency"],
            ),
        )

        out = out[np.argsort(out.time.data)]

    # Time clip
    if out is not None:
        out = time_clip(out, tint)

    return out
