#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings
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
    # all-NaN table (no data): nothing to reject
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        q = np.nanquantile(data, q=outliers_q)

    data_clean = np.where(data < q, data, np.nan)

    return data_clean


def feeps_remove_sunlit_sectors(feeps_eyes, outliers_q: float = 0.99, i_en: int = 3):
    r"""
    Remove sunlit sectors for each eye in the all eyes FEEPS data product. This is
    needed because the FEEPS contaminated sectors csv files are not updated often enough
    and there is visible spintone contamination in the data at times between updates.

    The function removes sunlit sectors from the all FEEPS eyes data product by
    identifying the sectors that are sunlit and replacing them with NaN. The idea is to
    make tables of 24 eyes x 64 sectors, similar to the ones used in the FEEPS
    contaminated sectors csv files.

    The function calculates the average observed flux in the energy channel `i_en`
    for each eye-sector pair and places it in a 24x64 array. Then replaces with NaN
    the eye-sector pairs that have a value greater than the outliers_q percentile of
    the eye-sector pairs with data.

    Parameters
    ----------
    feeps_eyes : xarray.Dataset
        The all FEEPS eyes data product.
    outliers_q : float, optional
        The quantile to use for outlier rejection. The default is 0.99.
    i_en : int, optional
        Index of the energy channel used to find the sunlit sectors. The default
        is 3.

    Returns
    -------
    feeps_eyes_new : xarray.Dataset
        The all FEEPS eyes data product with sunlit sectors removed.

    Notes
    -----
    The eye-sector pairs without data (no sample, or only NaNs and zeros) are left
    out of the quantile. Survey data are labelled with 8 of the 64 sectors, so the
    quantile is taken over these 8 sectors of each eye.

    """

    feeps_data_vars = list(feeps_eyes.data_vars.keys())
    top_eyes_keys = [data_var for data_var in feeps_data_vars if "top" in data_var]
    bot_eyes_keys = [data_var for data_var in feeps_data_vars if "bot" in data_var]

    # Rows of the 24 x 64 table: top eyes then bottom eyes
    eyes_rows = {key: int(key.split("-")[1]) - 1 for key in top_eyes_keys}
    eyes_rows.update({key: int(key.split("-")[1]) - 1 + 12 for key in bot_eyes_keys})

    # Times in each spin sector, found once for all eyes
    spin_sectors = feeps_eyes.spinsectnum.data
    times_in_ss = [np.flatnonzero(spin_sectors == ss) for ss in range(64)]

    table = np.full((24, 64), np.nan)

    for key, row in eyes_rows.items():
        data = feeps_eyes[key].data[:, i_en]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for ss, idx in enumerate(times_in_ss):
                table[row, ss] = np.nanmean(data[idx], axis=0)

    # sectors without data (zero flux) are left out of the quantile
    table = np.where(table == 0, np.nan, table)
    table_clean = _reject_outliers(table, outliers_q=outliers_q)

    new_nans = np.isnan(table_clean) & ~np.isnan(table)

    feeps_eyes_new = deepcopy(feeps_eyes)

    # Mask all the new sunlit sectors of an eye at once
    for key, row in eyes_rows.items():
        sunlit_sectors = np.flatnonzero(new_nans[row])

        if sunlit_sectors.size:
            feeps_eyes_new[key] = xr.where(
                ~feeps_eyes_new["spinsectnum"].isin(sunlit_sectors),
                feeps_eyes_new[key],
                np.nan,
                keep_attrs=True,
            )

    return feeps_eyes_new
