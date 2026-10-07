#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import xarray as xr

from .db_get_ts import _db_get_ts_dict

# Local imports
from .feeps_active_eyes import feeps_active_eyes

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

data_units_keys = {
    "flux": "intensity",
    "counts": "counts",
    "cps": "count_rate",
    "mask": "sector_mask",
}


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


def _eye_cdf_name(tar_var, e_id, mms_id, active_eyes):
    # Name of the variable of the eye e_id ("top-1", "bottom-12", ...)
    var, data_units = _tokenize(tar_var)
    pref = f"epd_feeps_{var['tmmode']}_{var['lev']}_{var['dtype']}"

    if e_id.split("-")[0] not in ["top", "bottom"]:
        raise ValueError("Invalid format of eye id")

    suf, sensor_id = e_id.split("-")[0], int(e_id.split("-")[1])

    if sensor_id not in active_eyes[suf]:
        raise ValueError(f"Unactive eye {e_id}")

    return f"mms{mms_id:d}_{pref}_{suf}_{data_units}_sensorid_{sensor_id:d}"


def get_feeps_alleyes(
    tar_var,
    tint,
    mms_id,
    verbose: bool = True,
    data_path: str = "",
    source: str = "default",
):
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
    source : {"default", "local", "sdc", "aws"}, Optional
        Resource to fetch the data from. Default uses default in
        `pyrfu/mms/config.json`. Each file is read (downloaded) once for all the
        eyes.

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
    # data_unit = tar_var.split("_")[0][:-1].lower()
    specie = tar_var.split("_")[0][-1]

    var = {
        "tmmode": tar_var.split("_")[1],
        "lev": tar_var.split("_")[2],
        "mmsId": mms_id,
        "units_name": tar_var.split("_")[0][:-1].lower(),
    }

    if specie == "e":
        var["dtype"] = "electron"
        var["species"] = "e"
    elif specie == "i":
        var["dtype"] = "ion"
        var["species"] = "i"
    else:
        raise ValueError("Invalid specie")

    dset_name = f"mms{mms_id:d}_feeps_{var['tmmode']}_l2_{var['dtype']}"
    pref = f"epd_feeps_{var['tmmode']}_{var['lev']}_{var['dtype']}"

    active_eyes = feeps_active_eyes(var, tint, mms_id)

    e_ids = [f"{k}-{s:d}" for k in active_eyes for s in active_eyes[k]]

    # Read all the variables at once (one download per file from the SDC or AWS)
    cdf_names = {
        "spinsectnum": f"mms{mms_id:d}_{pref}_spinsectnum",
        "pitch_angle": f"mms{mms_id:d}_{pref}_pitch_angle",
        **{e_id: _eye_cdf_name(tar_var, e_id, mms_id, active_eyes) for e_id in e_ids},
    }
    data = _db_get_ts_dict(
        dset_name, list(cdf_names.values()), tint, verbose, data_path, source
    )
    out_dict = {key: data[cdf_name] for key, cdf_name in cdf_names.items()}

    for e_id in e_ids:
        eye = out_dict[e_id]
        eye.attrs["tmmode"] = var["tmmode"]
        eye.attrs["lev"] = var["lev"]
        eye.attrs["mms_id"] = mms_id
        eye.attrs["dtype"] = var["dtype"]
        eye.attrs["species"] = f"{var['dtype']}s"
        out_dict[e_id] = eye.rename({eye.dims[1]: f"energy_{e_id}"})

    out = xr.Dataset(out_dict)

    out.attrs = var
    # out.attrs["data_units"]
    # out.attrs["species"] = var["dtype"]

    return out
