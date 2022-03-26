#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pdb
# 3rd party imports
import xarray as xr

# Local imports
from .feeps_active_eyes import feeps_active_eyes
from .db_get_ts import db_get_ts

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"

data_units_keys = {"flux": "intensity", "counts": "counts",
                   "cps": "count_rate", "mask": "sector_mask"}


def _tokenize(tar_var):
    var = {"inst": "feeps"}

    data_units = data_units_keys[tar_var.split("_")[0][:-1].lower()]

    specie = tar_var.split("_")[0][-1]

    if specie == "e":
        var["dtype"] = "electron"
    elif specie == "i":
        var["dtype"] = "ion"
    else:
        raise ValueError("invalid specie")

    var["tmmode"] = tar_var.split("_")[1]
    var["lev"] = tar_var.split("_")[2]

    return var, data_units


def _get_oneeye(tar_var, e_id, tint, mms_id, verbose: bool = True,
                data_path: str = ""):
    mms_id = int(mms_id)

    var, data_units = _tokenize(tar_var)

    dset_name = f"mms{mms_id:d}_feeps_{var['tmmode']}_l2_{var['dtype']}"
    pref = f"epd_feeps_{var['tmmode']}_{var['lev']}_{var['dtype']}"

    active_eyes = feeps_active_eyes(var, tint, mms_id)

    if e_id.split("-")[0] in ["top", "bottom"]:
        suf = e_id.split("-")[0]
        e_id = int(e_id.split("-")[1])

        assert e_id in active_eyes[suf], "Unactive eye"

        suf = f"{suf}_{data_units}_sensorid_{e_id:d}"

    else:
        raise ValueError("Invalid format of eye id")

    out = db_get_ts(dset_name, f"mms{mms_id:d}_{pref}_{suf}", tint, verbose,
                    data_path=data_path)

    out.attrs["tmmode"] = var["tmmode"]
    out.attrs["lev"] = var["lev"]
    out.attrs["mms_id"] = mms_id
    out.attrs["dtype"] = var["dtype"]
    out.attrs["species"] = "{}s".format(var["dtype"])
    return out


def get_feeps_alleyes(tar_var, tint, mms_id, verbose: bool = True,
                      data_path: str = ""):
    r"""Read energy spectrum of the selected specie in the selected energy
    range for all FEEPS eyes.

    Parameters
    ----------
    tar_var : str
        Key of the target variable like
        {data_unit}{specie}_{data_rate}_{data_lvl}.
    tint : list of str
        Time interval.
    mms_id : int or float or str
        Index of the spacecraft.
    verbose : bool, Optional
        Set to True to follow the loading. Default is True.
    data_path : str, Optional
        Path of MMS data. Default uses `pyrfu.mms.mms_config.py`

    Returns
    -------
    out : xarray.Dataset
        Dataset containing the energy spectrum of the available eyes of the
        Fly's Eye Energetic Particle Spectrometer (FEEPS).

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint_brst = ["2017-07-23T16:54:24.000", "2017-07-23T17:00:00.000"]

    Read electron energy spectrum for all FEEPS eyes

    >>> feeps_all_eyes = mms.get_feeps_alleyes("fluxe_brst_l2", tint_brst, 2)

    """

    mms_id = int(mms_id)

    specie = tar_var.split("_")[0][-1]

    var = {"tmmode": tar_var.split("_")[1], "lev": tar_var.split("_")[2],
           "mmsId": mms_id}

    if specie == "e":
        var["dtype"] = "electron"
    elif specie == "i":
        var["dtype"] = "ion"
    else:
        raise ValueError("Invalid specie")

    dset_name = f"mms{mms_id:d}_feeps_{var['tmmode']}_l2_{var['dtype']}"
    pref = f"epd_feeps_{var['tmmode']}_{var['lev']}_{var['dtype']}"

    active_eyes = feeps_active_eyes(var, tint, mms_id)

    e_ids = [f"{k}-{s:d}" for k in active_eyes for s in active_eyes[k]]

    out_dict = {"spinsectnum": db_get_ts(dset_name,
                                         f"mms{mms_id:d}_{pref}_spinsectnum",
                                         tint, data_path=data_path),
                "pitch_angle": db_get_ts(dset_name,
                                         f"mms{mms_id:d}_{pref}_pitch_angle",
                                         tint, data_path=data_path)}

    for e_id in e_ids:
        out_dict[e_id] = _get_oneeye(tar_var, e_id, tint, mms_id, verbose,
                                     data_path=data_path)
        dims = {o: n for o, n in zip(out_dict[e_id].dims, ["time", "energy"])}
        out_dict[e_id] = out_dict[e_id].rename(dims)

    out = xr.Dataset(out_dict)

    out.attrs = var

    out.attrs["specie"] = var["dtype"]

    return out
