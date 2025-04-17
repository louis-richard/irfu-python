#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Apostolos Kolokotronis"
__email__ = "apostolosk@irf.se"
__copyright__ = "Copyright 2020-2025"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _reject_outliers(data, outliers_q=0.99):

    q = np.nanquantile(data, q=outliers_q)
    data_clean = np.where(data < q, data, np.nan)

    return data_clean


def feeps_remove_sunlit_sectors(feeps_eyes, outliers_q=0.99):
    r"""
    Remove sunlit sectors for each eye in the all eyes FEEPS data product. This is
    needed because the FEEPS contaminated sectors csv files are not updated often enough
    and their is visible spintone contamination in the data at times between updates.

    The function removes sunlit sectors from the all FEEPS eyes data product by
    identifying the sectors that are sunlit and replacing them with NaN. The idea is to
    make tables of 24 eyes x 64 sectors, similar to the ones used in the FEEPS
    contaminated sectors csv files.

    The function calculates the average observed flux for each eye-sector pair and
    places it in a 24x64 array. Then replaces with NaN the eye-sector pairs that have
    a value greater than the outliers_q percentile of the data.

    Parameters
    ----------
    feeps_eyes : xarray.Dataset
        The all FEEPS eyes data product.
    outliers_q : float, optional
        The quantile to use for outlier rejection. The default is 0.99.

    Returns
    -------
    feeps_eyes_new : xarray.Dataset
        The all FEEPS eyes data product with sunlit sectors removed.
    """

    feeps_data_vars = list(feeps_eyes.data_vars.keys())
    i_en = 3
    top_eyes_keys = [data_var for data_var in feeps_data_vars if "top" in data_var]
    bot_eyes_keys = [data_var for data_var in feeps_data_vars if "bot" in data_var]

    table = np.empty(shape=(24, 64)) * np.nan

    for key in top_eyes_keys:
        key_id = int(key.split("-")[1])

        for ss in np.arange(0, 64):

            times_in_ss = feeps_eyes.where(
                feeps_eyes.spinsectnum == ss,
                drop=True,
            ).time
            data_in_ss = feeps_eyes[key].sel(time=times_in_ss).data[:, i_en]
            data_in_ss = np.nansum(data_in_ss, axis=0)
            table[key_id - 1, ss] = data_in_ss

    for key in bot_eyes_keys:
        key_id = int(key.split("-")[1])

        for ss in np.arange(0, 64):

            times_in_ss = feeps_eyes.where(
                feeps_eyes.spinsectnum == ss,
                drop=True,
            ).time
            data_in_ss = feeps_eyes[key].sel(time=times_in_ss).data[:, i_en]
            data_in_ss = np.nansum(data_in_ss, axis=0)
            table[key_id - 1 + 12, ss] = data_in_ss

    table_clean = np.empty_like(table) * np.nan
    table_clean = _reject_outliers(table, outliers_q=outliers_q)

    table = np.where(table == 0, np.nan, table)
    table_clean = np.where(table_clean == 0, np.nan, table_clean)

    nans = np.isnan(table)
    nans_clean = np.isnan(table_clean)

    new_nans = nans_clean & ~nans

    new_nans_indices = np.argwhere(new_nans)

    new_cs = new_nans_indices + np.array([1, 0])
    feeps_eyes_new = deepcopy(feeps_eyes)

    for cs in new_cs:

        eye_id = int(cs[0])
        ss = int(cs[1])

        if eye_id <= 12:

            feeps_eyes_new[f"top-{eye_id:d}"] = xr.where(
                feeps_eyes_new["spinsectnum"] != ss,
                feeps_eyes_new[f"top-{eye_id:d}"],
                np.nan,
                keep_attrs=True,
            )
        else:
            feeps_eyes_new[f"bottom-{eye_id - 12:d}"] = xr.where(
                feeps_eyes_new["spinsectnum"] != ss,
                feeps_eyes_new[f"bottom-{eye_id - 12:d}"],
                np.nan,
                keep_attrs=True,
            )

    return feeps_eyes_new
