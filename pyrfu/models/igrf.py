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
import warnings
import numpy as np
import pandas as pd

from scipy import interpolate
from astropy.time import Time


def igrf(time, flag):
    """Returns magnetic dipole latitude and longitude of the IGRF model

    Parameters
    ----------
    time : ndarray
        Times in unix format.

    flag : str
        Default is dipole.


    Returns
    -------
    lambda : ndarray
        latitude

    phi : ndarray
        longitude

    """

    # Root path
    # root_path = os.getcwd()
    root_path = os.path.dirname(os.path.abspath(__file__))

    # File path
    file_name = "igrf13coeffs.csv"

    # file reading
    df = pd.read_csv(os.sep.join([root_path, file_name]))

    # construct IGRF coefficient matrices
    a = df.loc[0]
    years_igrf = a[3:].to_list()
    years_igrf[-1] = float(years_igrf[-1].split("-")[0]) + 5.

    # read in all IGRF coefficients from file
    i_igrf = df.loc[1:, ].values
    i_igrf[:, -1] = i_igrf[:, -2] + 5. * i_igrf[:, -1].astype(float)
    h_igrf = i_igrf[i_igrf[:, 0] == "h", 1:].astype(float)
    g_igrf = i_igrf[i_igrf[:, 0] == "g", 1:].astype(float)

    # timeVec = irf_time(t,'vector');
    # yearRef = timeVec(:,1);
    year_ref = Time(time, format="unix").datetime64.astype("datetime64[Y]")
    year_ref = year_ref.astype(int) + 1970
    year_ref_isot = list(map(lambda x: f"{x}-01-01T00:00:00", year_ref))
    year_ref_unix = Time(year_ref_isot, format="isot").unix

    if np.min(year_ref) < np.min(years_igrf):
        message = "requested time is earlier than the first available IGRF " \
                  "model from extrapolating in past.. "
        warnings.warn(message, category=UserWarning)

    year = year_ref + (time - year_ref_unix) / (365.25 * 86400)

    assert flag == "dipole", "input flag is not recognized"

    tck_g0_igrf = interpolate.interp1d(years_igrf, g_igrf[0, 2:],
                                       kind="linear", fill_value="extrapolate")
    tck_g1_igrf = interpolate.interp1d(years_igrf, g_igrf[1, 2:],
                                       kind="linear", fill_value="extrapolate")
    tck_h0_igrf = interpolate.interp1d(years_igrf, h_igrf[0, 2:],
                                       kind="linear", fill_value="extrapolate")

    g01 = tck_g0_igrf(year)
    g11 = tck_g1_igrf(year)
    h11 = tck_h0_igrf(year)
    lambda_ = np.arctan(h11 / g11)
    phi = np.pi / 2
    phi -= np.arcsin((g11 * np.cos(lambda_) + h11 * np.sin(lambda_)) / g01)

    return np.rad2deg(lambda_), np.rad2deg(phi)

