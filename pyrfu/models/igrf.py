#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-on imports
import os
import warnings
from typing import Any, Optional, Tuple

# 3rd party imports
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import interpolate

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def igrf(
    time: NDArray[Any], flag: Optional[str] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Returns magnetic dipole latitude and longitude of the IGRF model

    Parameters
    ----------
    time : numpy.ndarray
        Times in unix format.
    flag : str
        Default is dipole.

    Returns
    -------
    tuple
        Tuple containing latitude and longitude as numpy arrays.

    Raises
    ------
    NotImplementedError
        If flag is not dipole.

    """

    if flag is None:
        flag = "dipole"

    # Root path
    # root_path = os.getcwd()
    path = os.sep.join(
        [os.path.dirname(os.path.abspath(__file__)), "igrf13coeffs.csv"],
    )
    df = pd.read_csv(path)

    # construct IGRF coefficient matrices
    years_igrf = df.loc[0][3:].to_list()
    years_igrf[-1] = float(years_igrf[-1].split("-")[0]) + 5.0

    # read in all IGRF coefficients from file
    i_igrf = df.loc[1:,].values
    i_igrf[:, -1] = i_igrf[:, -2] + 5.0 * i_igrf[:, -1].astype(np.float64)
    h_igrf = i_igrf[i_igrf[:, 0] == "h", 1:].astype(np.float64)
    g_igrf = i_igrf[i_igrf[:, 0] == "g", 1:].astype(np.float64)

    # timeVec = irf_time(t,'vector');
    # yearRef = timeVec(:,1);
    year_ref = (time * 1e9).astype("datetime64[ns]")
    year_ref = year_ref.astype("datetime64[Y]")
    year_ref = year_ref.astype(np.int64) + 1970
    year_ref_unix = (year_ref - 1970).astype("datetime64[Y]")
    year_ref_unix = year_ref_unix.astype("datetime64[ns]").astype(np.int64) / 1e9

    if np.min(year_ref) < np.min(years_igrf):
        warnings.warn(
            "requested time is earlier than the first available IGRF model; "
            "extrapolating in past",
            category=UserWarning,
        )

    if flag.lower() == "dipole":
        tck_g0_igrf = interpolate.interp1d(
            years_igrf,
            g_igrf[0, 2:],
            kind="linear",
            fill_value="extrapolate",
        )
        tck_g1_igrf = interpolate.interp1d(
            years_igrf,
            g_igrf[1, 2:],
            kind="linear",
            fill_value="extrapolate",
        )
        tck_h0_igrf = interpolate.interp1d(
            years_igrf,
            h_igrf[0, 2:],
            kind="linear",
            fill_value="extrapolate",
        )

        g01 = tck_g0_igrf(year_ref + (time - year_ref_unix) / (365.25 * 86400))
        g11 = tck_g1_igrf(year_ref + (time - year_ref_unix) / (365.25 * 86400))
        h11 = tck_h0_igrf(year_ref + (time - year_ref_unix) / (365.25 * 86400))
        lambda_ = np.arctan(h11 / g11)
        phi = np.pi / 2
        phi -= np.arcsin((g11 * np.cos(lambda_) + h11 * np.sin(lambda_)) / g01)
    else:
        raise NotImplementedError("input flag is not recognized")

    out = (np.rad2deg(lambda_), np.rad2deg(phi))
    return out
