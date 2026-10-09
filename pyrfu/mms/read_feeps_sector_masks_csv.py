#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import csv
import glob
import os

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf.iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def read_feeps_sector_masks_csv(tint):
    r"""Reads the FEEPS sectors to mask due to sunlight contamination from
    csv files.x

    Parameters
    ----------
    tint : list of str
        time range of interest [starttime, endtime] with the format
        "YYYY-MM-DD", "YYYY-MM-DD" or to specify more or less than a day [
        'YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']

    Returns
    -------
    mask : dict
        Hash table containing the sectors to mask for each spacecraft and
        sensor ID

    Notes
    -----
    The masks are read from the files in the ``sun`` folder dated nearest to
    the start time (UTC), as in IDL SPEDAS and pyspedas.

    """

    masks = {}

    # dates of the mask files shipped with pyrfu
    sun_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sun")
    files = glob.glob(os.path.join(sun_path, "MMS1_FEEPS_ContaminatedSectors_*.csv"))
    str_dates = sorted(os.path.basename(file)[-12:-4] for file in files)
    dates = np.array(
        [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in str_dates],
        dtype="datetime64[ns]",
    )

    # find the file closest to the start time
    t_start = iso86012datetime64(np.atleast_1d(tint[0]))[0]
    str_date = str_dates[np.argmin(np.abs(dates - t_start))]

    for mms_sc in np.arange(1, 5):
        file_name = f"MMS{mms_sc:d}_FEEPS_ContaminatedSectors_{str_date}.csv"
        csv_file = os.path.join(sun_path, file_name)

        # some files start with a byte order mark or end rows with a comma
        with open(csv_file, "r", encoding="utf-8-sig") as file:
            csv_data = [
                [float(x) for x in line if x.strip()] for line in csv.reader(file)
            ]

        csv_data = np.array(csv_data)

        for i in range(0, 12):
            mask_vals = []
            for val_idx in range(len(csv_data[:, i])):
                if csv_data[val_idx, i] == 1:
                    mask_vals.append(val_idx)

            masks[f"mms{mms_sc:d}_imask_top-{i + 1:d}"] = mask_vals

        for i in range(0, 12):
            mask_vals = []

            for val_idx in range(len(csv_data[:, i + 12])):
                if csv_data[val_idx, i + 12] == 1:
                    mask_vals.append(val_idx)

            masks[f"mms{mms_sc:d}_imask_bottom-{i + 1:d}"] = mask_vals

    return masks
