#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
get_feeps_oneeye.py

@author : Louis RICHARD
"""

import numpy as np

from .get_feeps_active_eyes import get_feeps_active_eyes
from .db_get_ts import db_get_ts


def get_feeps_oneeye(tar_var="fluxe_brst_l2", e_id="bottom-4", tint=None, mms_id=1, verbose=True):
    """
    Load energy spectrum all the target eye

    Parameters
    ----------
    tar_var : str
        target variable "{data_units}{specie}_{data_rate}_{level}" :
            * data_units :
                * flux : intensity (1/cm sr).
                * count : counts (-).
                * CPS : counts per second (1/s).

            * specie :
                * i : ion.
                * e : electron.

            * data_rate :
                * brst : high resolution data.
                * srvy : low resolution data.

            * level :
                * l1 : level 1 data
                * l1b : level 1b data
                * l2 : level 2 data
                * l3 : level 3 data

    e_id : str
        index of the eye "{deck}-{id}" :
            * deck : top/bottom
            * id : see get_feeps_active_eyes

    tint : list of str
        Time interval.

    mms_id : int or str
        Index of the spacecraft.

    verbose : bool, optional
        Set to True to follow the loading. Default is True.

    Returns
    -------
    out : xarray.DataArray
        Energy spectrum of the target eye.

    """

    assert isinstance(tar_var, str)
    assert isinstance(e_id, str)
    assert tint is not None and isinstance(tint, list)
    assert isinstance(tint[0], str) and isinstance(tint[1], str)
    assert isinstance(mms_id, (int, str)) and int(mms_id) in np.arange(1, 5)
    assert isinstance(verbose, bool)

    mms_id = int(mms_id)

    var = {"inst": "feeps"}

    data_units = tar_var.split("_")[0][:-1]
    specie = tar_var.split("_")[0][-1]

    if specie == "e":
        var["dtype"] = "electron"
    elif specie == "i":
        var["dtype"] = "ion"
    else:
        raise ValueError("invalid specie")

    var["tmmode"] = tar_var.split("_")[1]
    var["lev"] = tar_var.split("_")[2]

    dset_name = f"mms{mms_id:d}_feeps_{var['tmmode']}_l2_{var['dtype']}"
    dset_pref = f"mms{mms_id:d}_epd_feeps_{var['tmmode']}_{var['lev']}_{var['dtype']}"

    active_eyes = get_feeps_active_eyes(var, tint, mms_id)

    if e_id.split("-")[0] in ["top", "bottom"]:
        suf = e_id.split("-")[0]

        e_id = int(e_id.split("-")[1])

        if e_id in active_eyes[suf]:
            if data_units.lower() == "flux":
                suf = "_".join([suf, "intensity", "sensorid", str(e_id)])
            elif data_units.lower() == "counts":
                suf = "_".join([suf, "counts", "sensorid", str(e_id)])
            elif data_units.lower() == "cps":
                suf = "_".join([suf, "count_rate", "sensorid", str(e_id)])
            elif data_units == "mask":
                suf = "_".join([suf, "sector_mask", "sensorid", str(e_id)])
            else:
                raise ValueError("undefined variable")
        else:
            raise ValueError("Unactive eye")
    else:
        raise ValueError("Invalid format of eye id")

    if verbose:
        print("Loading {}...".format("_".join([dset_pref, suf])))

    out = db_get_ts(dset_name, "_".join([dset_pref, suf]), tint)

    out.attrs["tmmode"] = var["tmmode"]
    out.attrs["lev"] = var["lev"]
    out.attrs["mms_id"] = mms_id
    out.attrs["dtype"] = var["dtype"]
    out.attrs["species"] = "{}s".format(var["dtype"])
    return out
