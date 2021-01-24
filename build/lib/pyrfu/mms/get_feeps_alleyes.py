#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

import xarray as xr

from .get_feeps_oneeye import get_feeps_oneeye
from .get_feeps_active_eyes import get_feeps_active_eyes


def get_feeps_alleyes(tar_var, tint, mms_id, verbose=True):
    """Read energy spectrum of the selected specie in the selected energy range for all FEEPS eyes.

    Parameters
    ----------
    tar_var : str
        Key of the target variable like {data_unit}{specie}_{data_rate}_{data_lvl}.

    tint : list of str
        Time interval.

    mms_id : int or float or str
        Index of the spacecraft.

    verbose : bool
        Set to True to follow the loading. Default is True.

    Returns
    -------
    out : xarray.Dataset
        Dataset containing the energy spectrum of the available eyes of the Fly's Eye Energetic
        Particle Spectrometer.

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

    var = {"tmmode": tar_var.split("_")[1], "lev": tar_var.split("_")[2], "mmsId": mms_id}

    if specie == "e":
        var["dtype"] = "electron"
    elif specie == "i":
        var["dtype"] = "ion"
    else:
        raise ValueError("Invalid specie")

    active_eyes = get_feeps_active_eyes(var, tint, mms_id)

    e_ids = [f"{k}-{s:d}" for k in active_eyes for s in active_eyes[k]]

    out_dict = {}

    for e_id in e_ids:
        out_dict[e_id] = get_feeps_oneeye(tar_var, e_id, tint, mms_id, verbose)

    out = xr.Dataset(out_dict)

    out.attrs = var

    out.attrs["species"] = "{}s".format(var["dtype"])

    return out
