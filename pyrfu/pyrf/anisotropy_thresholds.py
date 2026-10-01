#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Dict, Union

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2025"
__license__ = "MIT"
__version__ = "2.4.14"
__status__ = "Prototype"

# Coefficients for electron instabilities
COEFFS_E: Dict[float, Dict[str, tuple]] = {
    0.01: {
        "firehose": (-1.23, 0.88),
        "whistler": (0.36, 0.55),
    },
    0.1: {
        "firehose": (-1.32, 0.61),
        "whistler": (1.0, 0.49),
    },
}

# Coefficients for ion instabilities
COEFFS_I: Dict[float, Dict[str, tuple]] = {
    0.01: {
        "proton cyclotron": (0.649, 0.400, 0.0),
        "mirror mode": (1.040, 0.633, -0.012),
        "parallel firehose": (-0.647, 0.583, 0.713),
        "oblique firehose": (-1.447, 1.000, -0.148),
    },
    0.001: {
        "proton cyclotron": (0.437, 0.428, -0.003),
        "mirror mode": (0.801, 0.763, -0.063),
        "parallel firehose": (-0.497, 0.566, 0.543),
        "oblique firehose": (-1.390, 1.005, -0.111),
    },
    0.0001: {
        "proton cyclotron": (0.367, 0.364, 0.011),
        "mirror mode": (0.702, 0.674, -0.009),
        "parallel firehose": (-0.408, 0.529, 0.410),
        "oblique firehose": (-1.454, 1.023, -0.178),
    },
}


def _thresh_e(beta_para: np.ndarray, s: float, alpha: float) -> np.ndarray:
    # 1 + s * beta_para ** -alpha, NaN for beta_para <= 0 (undefined). The input
    # is not modified, as it is shared by all the instabilities.
    beta_pos = np.where(beta_para > 0, beta_para, np.nan)
    t_aniso = 1 + s * beta_pos**-alpha
    return t_aniso


def _thresh_i(beta_para: np.ndarray, a: float, b: float, beta0: float) -> np.ndarray:
    # 1 + a / (beta_para - beta0) ** b, NaN for beta_para <= beta0 (undefined).
    # The input is not modified, as it is shared by all the instabilities.
    d_beta = np.where(beta_para > beta0, beta_para - beta0, np.nan)
    t_aniso = 1 + a / d_beta**b
    return t_aniso


def anisotropy_thresholds(
    beta_para: Union[float, np.ndarray, xr.DataArray],
    specie: str = "i",
    gamma: float = 0.01,
) -> Dict[str, Union[float, np.ndarray, xr.DataArray]]:
    r"""Compute the thresholds for temperature anisotropy instabilities based on
    plasma species and growth rate.

    Parameters
    ----------
    beta_para : float or array_like or xarray.DataArray
        Parallel beta. It is not modified.
    specie : str, optional
        Plasma species, "i" for ions or "e" for electrons. Default is "i".
    gamma : float, optional
        Growth rate of the instability. Must match a key in the corresponding
        coefficient dictionary. Default is 0.01.

    Returns
    -------
    dict
        Thresholds of the temperature anisotropy T_perp / T_para, with the
        instability names as keys, of the same type as `beta_para` (float,
        numpy.ndarray, or xarray.DataArray with the same coordinates). NaN where
        the fit is undefined (beta_para <= beta0 for ions, beta_para <= 0 for
        electrons) and for negative beta_para.

    Raises
    ------
    ValueError
        If specie is not "i" or "e", or if gamma is not supported.

    Notes
    -----
    The thresholds are fits of the form T_perp / T_para = 1 + a / (beta_para -
    beta0) ** b for ions, and 1 + s / beta_para ** alpha for electrons. The
    firehose thresholds become negative at low beta_para, where the fits don't
    apply.

    """

    # Copy as floats, so that the input is not modified, and mask the negative
    # values to avoid invalid calculations
    beta = np.array(beta_para, dtype=np.float64)
    beta = np.where(beta < 0, np.nan, beta)

    if specie == "i":
        if gamma not in COEFFS_I:
            gammas = list(COEFFS_I.keys())
            raise ValueError(
                f"Unsupported gamma value {gamma} for ions. Available: {gammas}"
            )
        coeffs = COEFFS_I[gamma]
        out = {name: _thresh_i(beta, *params) for name, params in coeffs.items()}

    elif specie == "e":
        if gamma not in COEFFS_E:
            gammas = list(COEFFS_E.keys())
            raise ValueError(
                f"Unsupported gamma value {gamma} for electrons. Available: {gammas}"
            )
        coeffs = COEFFS_E[gamma]
        out = {name: _thresh_e(beta, *params) for name, params in coeffs.items()}

    else:
        raise ValueError(f"Unknown specie '{specie}'. Expected 'i' or 'e'.")

    # Same type as the input
    if isinstance(beta_para, xr.DataArray):
        out = {
            name: xr.DataArray(
                value, coords=beta_para.coords, dims=beta_para.dims, name=name
            )
            for name, value in out.items()
        }
    elif np.ndim(beta_para) == 0:
        out = {name: float(value) for name, value in out.items()}

    return out
