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

import os
import csv
import numpy as np

from astropy.time import Time


def read_feeps_sector_masks_csv(tint):
    """
    Reads the FEEPS sectors to mask due to sunlight contamination from csv files.x

    Parameters
    ----------
    tint : list of str
        time range of interest [starttime, endtime] with the format "YYYY-MM-DD", "YYYY-MM-DD"]
        or to specify more or less than a day ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']

    Returns
    -------
    mask : dict
        Hash table containing the sectors to mask for each spacecraft and sensor ID

    """

    masks = {}

    dates = [1447200000.0000000,  # 11/11/2015
             1468022400.0000000,  # 7/9/2016
             1477612800.0000000,  # 10/28/2016
             1496188800.0000000,  # 5/31/2017
             1506988800.0000000,  # 10/3/2017
             1538697600.0000000]  # 10/5/2018

    # find the file closest to the start time
    nearest_date = dates[(np.abs(np.array(dates)-Time(tint[0], format="isot").unix)).argmin()]
    str_date = Time(nearest_date, format="unix").datetime.strftime("%Y%m%d")

    for mms_sc in np.arange(1, 5):
        csv_file = os.sep.join([os.path.dirname(os.path.abspath(__file__)), "sun",
                                "MMS{:d}_FEEPS_ContaminatedSectors_{}.csv".format(mms_sc,
                                                                                  str_date)])

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

            masks["mms{:d}_imask_top-{:d}".format(mms_sc, i + 1)] = mask_vals

        for i in range(0, 12):
            mask_vals = []

            for val_idx in range(len(csv_data[:, i+12])):
                if csv_data[val_idx, i+12] == 1:
                    mask_vals.append(val_idx)

            masks["mms{:d}_imask_bottom-{:d}".format(mms_sc, i + 1)] = mask_vals

    return masks
