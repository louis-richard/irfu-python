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

import numpy as np
import xarray as xr

from astropy.time import Time

from .db_get_ts import db_get_ts
from .read_feeps_sector_masks_csv import read_feeps_sector_masks_csv


def feeps_remove_sun(inp_dataset):
    """Removes the sunlight contamination from FEEPS data.

    Parameters
    ----------
    inp_dataset : xarray.Dataset
        Dataset of energy spectrum of all eyes.

    Returns
    -------
    out : xarray.Dataset
        Dataset of cleaned energy spectrum of all eyes.

    See also
    --------
    pyrfu.mms.get_feeps_alleyes : Read energy spectrum for all FEEPS eyes.

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint = ["2017-07-18T13:04:00.000", "2017-07-18T13:07:00.000"]

    Spacecraft index

    >>> mms_id = 2

    Load data from FEEPS

    >>> cps_i = mms.get_feeps_alleyes("CPSi_brst_l2", tint, mms_id)
    >>> cps_i_clean, _ = mms.feeps_split_integral_ch(cps_i)
    >>> cps_i_clean_sun_removed = mms.feeps_remove_sun(cps_i_clean)

    """

    var = inp_dataset.attrs

    trange = list(Time(inp_dataset.time.data[[0, -1]], format="datetime64").isot)

    dataset_name = "mms{:d}_feeps_{}_{}_{}".format(var["mmsId"], var["tmmode"], var["lev"],
                                                   var["dtype"])
    dataset_pref = "mms{:d}_epd_feeps_{}_{}_{}".format(var["mmsId"], var["tmmode"], var["lev"],
                                                       var["dtype"])

    spin_sectors = db_get_ts(dataset_name, "_".join([dataset_pref, "spinsectnum"]), trange)
    mask_sectors = read_feeps_sector_masks_csv(trange)

    out_dict = {}

    for k in inp_dataset:
        out_dict[k] = inp_dataset[k]
        if mask_sectors.get("mms{:d}_imask_{}".format(var["mmsId"], k)) is not None:
            bad_sectors = mask_sectors["mms{:d}_imask_{}".format(var["mmsId"], k)]

            for bad_sector in bad_sectors:
                this_bad_sector = np.where(spin_sectors == bad_sector)[0]
                if len(this_bad_sector) != 0:
                    out_dict[k].data[this_bad_sector] = np.nan

    out = xr.Dataset(out_dict, attrs=var)

    out.attrs = var

    return out
