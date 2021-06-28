#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-on imports
import os
import warnings

# 3rd party imports
import numpy as np
import pandas as pd

from scipy import interpolate


def igrf(time, flag):
    r"""Returns magnetic dipole latitude and longitude of the IGRF model

    Parameters
    ----------
    time : numpy.ndarray
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
    path = os.sep.join([os.path.dirname(os.path.abspath(__file__)),
                             "igrf13coeffs.csv"])
    df = pd.read_csv(path)

    # construct IGRF coefficient matrices
    years_igrf = df.loc[0][3:].to_list()
    years_igrf[-1] = float(years_igrf[-1].split("-")[0]) + 5.

    # read in all IGRF coefficients from file
    i_igrf = df.loc[1:, ].values
    i_igrf[:, -1] = i_igrf[:, -2] + 5. * i_igrf[:, -1].astype(float)
    h_igrf = i_igrf[i_igrf[:, 0] == "h", 1:].astype(float)
    g_igrf = i_igrf[i_igrf[:, 0] == "g", 1:].astype(float)

    # timeVec = irf_time(t,'vector');
    # yearRef = timeVec(:,1);
    year_ref = (time * 1e9).astype("datetime64[ns]")
    year_ref = year_ref.astype("datetime64[Y]")
    year_ref = year_ref.astype(int) + 1970
    year_ref_unix = (year_ref - 1970).astype("datetime64[Y]")
    year_ref_unix = year_ref_unix.astype("datetime64[ns]").astype(int) / 1e9

    if np.min(year_ref) < np.min(years_igrf):
        message = "requested time is earlier than the first available IGRF " \
                  "model from extrapolating in past.. "
        warnings.warn(message, category=UserWarning)

    assert flag == "dipole", "input flag is not recognized"

    tck_g0_igrf = interpolate.interp1d(years_igrf, g_igrf[0, 2:],
                                       kind="linear", fill_value="extrapolate")
    tck_g1_igrf = interpolate.interp1d(years_igrf, g_igrf[1, 2:],
                                       kind="linear", fill_value="extrapolate")
    tck_h0_igrf = interpolate.interp1d(years_igrf, h_igrf[0, 2:],
                                       kind="linear", fill_value="extrapolate")

    g01 = tck_g0_igrf(year_ref + (time - year_ref_unix) / (365.25 * 86400))
    g11 = tck_g1_igrf(year_ref + (time - year_ref_unix) / (365.25 * 86400))
    h11 = tck_h0_igrf(year_ref + (time - year_ref_unix) / (365.25 * 86400))
    lambda_ = np.arctan(h11 / g11)
    phi = np.pi / 2
    phi -= np.arcsin((g11 * np.cos(lambda_) + h11 * np.sin(lambda_)) / g01)

    return np.rad2deg(lambda_), np.rad2deg(phi)
