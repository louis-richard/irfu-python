#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import csv

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf import datetime642unix, unix2datetime64, iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
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

    """

    masks = {}

    dates = [1447200000.0000000,  # 11/11/2015
             1468022400.0000000,  # 7/9/2016
             1477612800.0000000,  # 10/28/2016
             1496188800.0000000,  # 5/31/2017
             1506988800.0000000,  # 10/3/2017
             1538697600.0000000]  # 10/5/2018

    # find the file closest to the start time
    date = datetime642unix(iso86012datetime64(np.array(tint)[0]))
    nearest_date = dates[np.argmin((np.abs(np.array(dates) - date)))]
    nearest_date = unix2datetime64(np.array(nearest_date))
    str_date = nearest_date.astype("<M8[D]").astype(str).replace("-", "")

    for mms_sc in np.arange(1, 5):
        file_name = f"MMS{mms_sc:d}_FEEPS_ContaminatedSectors_{str_date}.csv"
        csv_file = os.sep.join([os.path.dirname(os.path.abspath(__file__)),
                                "sun", file_name])

        csv_file = open(csv_file, 'r')

        csv_reader = csv.reader(csv_file)

        csv_data = []

        for line in csv_reader:
            csv_data.append([float(l) for l in line])

        csv_file.close()

        csv_data = np.array(csv_data)

        for i in range(0, 12):
            mask_vals = []
            for val_idx in range(len(csv_data[:, i])):
                if csv_data[val_idx, i] == 1:
                    mask_vals.append(val_idx)

            masks[f"mms{mms_sc:d}_imask_top-{i + 1:d}"] = mask_vals

        for i in range(0, 12):
            mask_vals = []

            for val_idx in range(len(csv_data[:, i+12])):
                if csv_data[val_idx, i+12] == 1:
                    mask_vals.append(val_idx)

            masks[f"mms{mms_sc:d}_imask_bottom-{i + 1:d}"] = mask_vals

    return masks
