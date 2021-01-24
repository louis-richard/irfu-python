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


def get_feeps_energy_table(mms_id, eye, sensor_id):
    """This function returns the energy table based on each spacecraft and eye; based on the
    table from : FlatFieldResults_V3.xlsx
    
    Parameters
    ----------
    mms_id : str
        Spacecraft index e.g., "4" for MMS4.

    eye : str
        Sensor eye #.

    sensor_id : int
        Sensor ID.

    Returns
    -------
    out : list
        Energy table.
        
    Notes
    -----
    Bad eyes are replaced by NaNs.
    Different original energy tables are used depending on if the sensor head is 6-8 (ions) or not
    (electrons) :

        * Electron Eyes: 1, 2, 3, 4, 5, 9, 10, 11, 12
        * Ion Eyes: 6, 7, 8

    """

    table = {"mms1-top": [14., 7., 16., 14., 14., 0., 0., 0., 14., 14., 17., 15.],
             "mms1-bot": [np.nan, 14., 14., 13., 14., 0., 0., 0., 14., 14., -25., 14.],
             "mms2-top": [-1., 6., -2., -1., np.nan, 0., np.nan, 0., 4., -1., -1., 0.],
             "mms2-bot": [-2., -1., -2., 0., -2., 15., np.nan, 15., -1., -2., -1., -3.],
             "mms3-top": [-3., np.nan, 2., -1., -5., 0., 0., 0., -3., -1., -3., np.nan],
             "mms3-bot": [-7., np.nan, -5., -6., np.nan, 0., 0., 12., 0., -2., -3., -3.],
             "mms4-top": [np.nan, np.nan, -2., -5., -5., 0., np.nan, 0., -1., -3., -6., -6.],
             "mms4-bot": [-8., np.nan, -2., np.nan, np.nan, -8., 0., 0., -2., np.nan, np.nan, -4.]}

    if 6 <= sensor_id <= 8:
        mms_energies = [57.90, 76.80, 95.40, 114.1, 133.0, 153.7, 177.6, 205.1, 236.7, 273.2, 315.4,
                        363.8, 419.7, 484.2, 558.6, 609.9]
    else:
        mms_energies = [33.20, 51.90, 70.60, 89.40, 107.1, 125.2, 146.5, 171.3, 200.2, 234.0, 273.4,
                        319.4, 373.2, 436.0, 509.2, 575.8]

    out = [energy + table[f"mms{mms_id}-{eye}"][sensor_id - 1] for energy in mms_energies]

    return out
