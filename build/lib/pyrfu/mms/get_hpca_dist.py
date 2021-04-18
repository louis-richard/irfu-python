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

"""get_hpca_dist.py
@author: Louis Richard
"""

from ..pyrf import ts_append

from .list_files import list_files
from .get_ts import get_ts


def get_hpca_dist(var_key, tint, mms_id, data_path: str = ""):
    r"""Loads Hot Plasma Composition Analyser (HPCA) full distribution.

    Parameters
    ----------
    var_key : str
        key like "{flux type}_{specie}_{time mode}_{level}".

    tint : list
        Time interval.

    mms_id : int
        Spacecraft index.

    data_path : str, optional
        Path to data


    Returns
    -------
    vdf_ : xarray.DataArray
        Distribution PSD or DPF.

    saz_ : xarray.DataArray
        Start azimuthal angle.

    aze_ : xarray.DataArray
        Azimuthal angles.

    """

    var = {"inst": "hpca", "dtype": "ion"}
    flux, specie, var["tmmode"], var["lev"] = var_key.split("_")

    files = list_files(tint, mms_id, var, data_path)

    if flux == "psd":
        flux = "phase_space_density"
    elif flux == "dpf":
        flux = "flux"
    else:
        raise ValueError("Invalid flux type")

    prefix_ = f"mms{mms_id}_hpca"
    suf_vdf = f"{specie}_{flux}"
    suf_aze = "azimuth_angles_per_ev_degrees"
    suf_saz = f"start_azimuth"

    vdf_, saz_, aze_ = [None, None, None]

    for file in files:
        vdf_ = ts_append(vdf_, get_ts(file, f"{prefix_}_{suf_vdf}", tint))
        saz_ = ts_append(saz_, get_ts(file, f"{prefix_}_{suf_saz}", tint))
        aze_ = ts_append(aze_, get_ts(file, f"{prefix_}_{suf_aze}", tint))

    return vdf_, saz_, aze_
