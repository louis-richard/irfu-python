#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import numbers
import os
from typing import Optional

# 3rd party imports
import numpy as np
import pycdfpp
import xarray as xr
from xarray import DataArray

from ..pyrf import read_cdf, time_clip
from ._list_files import _list_files

logger = logging.getLogger(__name__)


def _list_files_tnr_l2(
    tint: list, data_path: Optional[str] = "", tree: Optional[bool] = False
) -> list:
    """Latest version of the L2_rpw-tnr-surv files covering the time interval."""
    return _list_files(tint, "L2_rpw-tnr-surv", ["L2", "thr"], data_path, tree)


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
    TypeError
        If sensor is not an integer, tint not two strings or tree not a bool.
    FileNotFoundError
        If the data path doesn't exist, or no file covers the time interval.
    ValueError
        If tint has not two elements, or no file has data from the sensor
        selected.

    Notes
    -----
    As irfu-matlab solo.read_TNR, the two channels are put together, one
    spectrum per sweep, the bands not measured in a sweep are NaN and the
    sweeps without positive power are skipped. The files without data from
    the sensor are skipped.

    """

    # Check input types (numpy integers, e.g. from an array, were rejected)
    if not isinstance(sensor, numbers.Integral):
        raise TypeError("sensor must be an integer")

    files = _list_files_tnr_l2(tint, data_path, tree)

    # (None was returned)
    if not files:
        raise FileNotFoundError(f"no L2 TNR file in {tint}")

    # Spectra of each file, put together once at the end (ts_append in the
    # loop copied all the previous files for each one)
    spectra = []

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
        sweep_num = sweep_num[sens_]

        # One spectrum per sweep, the sweeps in increasing order; each record
        # fills its band (32 frequencies of 128), a later record of the same
        # band and sweep replacing an earlier one. All the sweeps at once (a
        # search of the records of each sweep, quadratic in their number)
        sweeps, first, which = np.unique(
            sweep_num, return_index=True, return_inverse=True
        )
        which = np.ravel(which)

        # The last record of each (sweep, band)
        keys = which * 4 + bande_e
        _, last = np.unique(keys[::-1], return_index=True)
        last = len(keys) - 1 - last

        v_all = np.zeros((len(sweeps), 4, 32))
        v_all[which[last], bande_e[last]] = auto_calib[last]
        v_all = v_all.reshape(len(sweeps), 128)

        # Sweeps without positive power (e.g., fill values) are skipped; the
        # bands not measured are NaN
        positive = np.sum(v_all, axis=1) > 0.0
        v_ = v_all[positive]
        v_[v_ == 0.0] = np.nan
        time = time_rr[first[positive]]

        if not v_.size:
            logger.info("no spectrum from sensor %d in %s", sensor, file)
            continue

        # Frequencies below 100 kHz
        f_100 = freq_tnr < 1e2

        spectra.append(
            xr.DataArray(
                v_[:, f_100],
                coords=[time, freq_tnr[f_100]],
                dims=["time", "frequency"],
            )
        )

    if not spectra:
        raise ValueError(f"no data from sensor {sensor} in {tint}")

    out = xr.concat(spectra, dim="time")
    out = out[np.argsort(out.time.data)]

    # Time clip
    out = time_clip(out, tint)

    return out
