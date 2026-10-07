#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import warnings
from functools import lru_cache
from typing import Any, Optional, Tuple

# 3rd party imports
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import interpolate

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


@lru_cache(maxsize=1)
def _igrf_coefficients() -> Tuple[NDArray[np.float64], ...]:
    # IGRF-14 coefficients, igrf14coeffs.txt of the IAGA with the comments
    # removed and the columns separated by commas. Read once, read-only.
    path = os.sep.join(
        [os.path.dirname(os.path.abspath(__file__)), "igrf14coeffs.csv"],
    )
    df = pd.read_csv(path, header=1)

    # Model epochs, the last column is the secular variation over the 5 years
    # following the last epoch
    years_igrf = df.columns[3:-1].astype(np.float64).to_numpy()
    years_igrf = np.append(years_igrf, years_igrf[-1] + 5.0)

    # the last column is the derivative, make it the value 5 years later
    coeffs = df.iloc[:, 1:].to_numpy(dtype=np.float64, copy=True)
    coeffs[:, -1] = coeffs[:, -2] + 5.0 * coeffs[:, -1]
    g_igrf = coeffs[df.iloc[:, 0].to_numpy() == "g"]
    h_igrf = coeffs[df.iloc[:, 0].to_numpy() == "h"]

    for array in (years_igrf, g_igrf, h_igrf):
        array.setflags(write=False)

    return years_igrf, g_igrf, h_igrf


def igrf(
    time: NDArray[Any], flag: Optional[str] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Returns magnetic dipole longitude and latitude of the IGRF-14 model [1]_

    Parameters
    ----------
    time : numpy.ndarray
        Times in unix format.
    flag : str
        Default is dipole.

    Returns
    -------
    tuple
        Tuple containing the longitude and the latitude of the dipole axis, in
        degrees, as numpy arrays.

    Raises
    ------
    NotImplementedError
        If flag is not dipole.

    Notes
    -----
    The coefficients are interpolated linearly between the model epochs
    (1900-2025), using the secular variation for 2025-2030, and extrapolated
    linearly outside with a warning. The dipole latitude uses the arctan of the
    Hapgood (1997) correction to Hapgood (1992), as irfu-matlab.

    References
    ----------
    .. [1]  IAGA Division V-MOD, International Geomagnetic Reference Field,
            14th generation (IGRF-14),
            https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field

    """

    if flag is None:
        flag = "dipole"

    years_igrf, g_igrf, h_igrf = _igrf_coefficients()

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

    # Decimal year, as irfu-matlab
    year = year_ref + (time - year_ref_unix) / (365.25 * 86400)

    if np.max(year) > np.max(years_igrf):
        warnings.warn(
            "requested time is later than the IGRF secular variation; "
            "extrapolating in future",
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

        g01 = tck_g0_igrf(year)
        g11 = tck_g1_igrf(year)
        h11 = tck_h0_igrf(year)
        lambda_ = np.arctan(h11 / g11)
        # Hapgood (1997) replaced the arcsin of Hapgood (1992) with an arctan
        phi = np.pi / 2
        phi -= np.arctan((g11 * np.cos(lambda_) + h11 * np.sin(lambda_)) / g01)
    else:
        raise NotImplementedError("input flag is not recognized")

    out = (np.rad2deg(lambda_), np.rad2deg(phi))
    return out
