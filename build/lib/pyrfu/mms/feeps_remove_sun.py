#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .read_feeps_sector_masks_csv import read_feeps_sector_masks_csv

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def feeps_remove_sun(inp_dataset):
    r"""Removes the sunlight contamination from FEEPS data.

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

    tint = list(np.datetime_as_string(inp_dataset.time.data[[0, -1]], "ns"))

    spin_sectors = inp_dataset["spinsectnum"]
    mask_sectors = read_feeps_sector_masks_csv(tint)

    out_dict = {}
    out_dict["spinsectnum"] = inp_dataset["spinsectnum"]

    for k in inp_dataset:
        out_dict[k] = inp_dataset[k]
        if mask_sectors.get(f"mms{var['mmsId']:d}_imask_{k}") is not None:
            bad_sectors = mask_sectors[f"mms{var['mmsId']:d}_imask_{k}"]

            for bad_sector in bad_sectors:
                this_bad_sector = np.where(spin_sectors == bad_sector)[0]
                if len(this_bad_sector) != 0:
                    out_dict[k].data[this_bad_sector] = np.nan

    out = xr.Dataset(out_dict, attrs=var)

    out.attrs = var

    return out
