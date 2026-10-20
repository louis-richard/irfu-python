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
from xarray.core.dataarray import DataArray

from ..pyrf import read_cdf, time_clip, ts_append
from .db_init import config_path

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

logger = logging.getLogger(__name__)


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
        Path of Solar Orbiter data. Default uses the local_data_dir of the
        configuration in the user directory (see pyrfu.solo.db_init).
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
        # Read the configuration file (user directory)
        with open(config_path(), "r", encoding="utf-8") as fs:
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
        Path of Solar Orbiter data. Default uses the local_data_dir of the
        configuration in the user directory (see pyrfu.solo.db_init).
    tree : bool, Optional
        Flag for tree structured data repos. Default is False.

    Returns
    -------
    DataArray
        Spectrum of the measured signals, below 100 kHz (frequencies in kHz).

    Raises
    ------
    ValueError
        If no file has data from the sensor selected.

    Notes
    -----
    As irfu-matlab solo.read_TNR, the two channels are put together, one
    spectrum per sweep, the bands not measured in a sweep are NaN and the
    sweeps without positive power are skipped. The files without data from
    the sensor are skipped.

    """

    # Check input types
    assert isinstance(sensor, int), "sensor must integer"

    files = _list_files_tnr_l2(tint, data_path, tree)

    # Initialize spectrum output to None
    out = None

    for file in files:
        # Notify user
        logger.info("Loading %s...", os.path.split(file)[-1])

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

        # Renumber the sweeps after a reset of SWEEP_NUM (a jump of more than
        # 100), as read_TNR.m, in int64: differences of the uint32 wrapped, so
        # that any small decrease (records out of order) counted as a reset
        sweep_num = sweep_.astype(np.int64)
        xdelta_sw = np.where(np.abs(np.diff(sweep_num)) > 100)[0]

        if xdelta_sw.size:
            xdelta_sw = np.hstack([xdelta_sw, len(sweep_num) - 1])

            for inswn in range(len(xdelta_sw) - 1):
                idx_l, idx_r = [xdelta_sw[inswn] + 1, xdelta_sw[inswn + 1]]
                sweep_num[idx_l : idx_r + 1] += sweep_num[xdelta_sw[inswn]]

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
            logger.info("no data from sensor %d in %s", sensor, file)
            continue

        ord_time = np.argsort(timet_ici)
        time_rr = timet_ici[ord_time]
        sens_ = sens_[ord_time]
        auto_calib = auto_calib[ord_time, :]

        bande_e = bande_[sens_]
        max_sweep = np.max(sweep_num[sens_])
        min_sweep = np.min(sweep_num[sens_])
        sweep_num = sweep_num[sens_]

        v_, time = [[], []]

        for ind_sweep in range(min_sweep, max_sweep + 1):
            p_punt = np.where(sweep_num == ind_sweep)[0]

            if not p_punt.size:
                continue

            v1_ = np.zeros(128)

            for idx in p_punt:
                v1_[32 * bande_e[idx] : 32 * (bande_e[idx] + 1)] = auto_calib[idx, :]

            # Sweeps without positive power (e.g., fill values) are skipped;
            # the bands not measured are NaN
            if np.sum(v1_) > 0.0:
                v1_[v1_ == 0.0] = np.nan
                v_.append(v1_)
                time.append(time_rr[np.min(p_punt)])

        if not v_:
            logger.info("no spectrum from sensor %d in %s", sensor, file)
            continue

        # Frequencies below 100 kHz
        f_100 = freq_tnr < 1e2

        out = ts_append(
            out,
            xr.DataArray(
                np.stack(v_)[:, f_100],
                coords=[np.stack(time), freq_tnr[f_100]],
                dims=["time", "frequency"],
            ),
        )

        out = out[np.argsort(out.time.data)]

    if files and out is None:
        raise ValueError(f"no data from sensor {sensor} in {tint}")

    # Time clip
    if out is not None:
        out = time_clip(out, tint)

    return out
