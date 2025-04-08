#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Dict, Union

# 3rd party imports
import numpy as np

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


def _thresh_e(
    beta_para: Union[float, np.ndarray], s: float, alpha: float
) -> Union[float, np.ndarray]:
    return 1 + s * beta_para**-alpha


def _thresh_i(
    beta_para: Union[float, np.ndarray], a: float, b: float, beta0: float
) -> Union[float, np.ndarray]:
    return 1 + a / (beta_para - beta0) ** b


def anisotropy_thresholds(
    beta_para: Union[float, np.ndarray], specie: str = "i", gamma: float = 0.01
) -> Dict[str, Union[float, np.ndarray]]:
    r"""Compute the thresholds for temperature anisotropy instabilities based on
    plasma species and growth rate.

    Parameters
    ----------
    beta_para : float or array_like
        Parallel beta.
    specie : str, optional
        Plasma species, "i" for ions or "e" for electrons. Default is "i".
    gamma : float, optional
        Growth rate of the instability. Must match a key in the corresponding
        coefficient dictionary. Default is 0.01.

    Returns
    -------
    dict
        Dictionary of thresholds with instability names as keys.

    Raises
    ------
    ValueError
        If specie is not "i" or "e", or if gamma is not supported.

    """

    if specie == "i":
        if gamma not in COEFFS_I:
            gammas = list(COEFFS_I.keys())
            raise ValueError(
                f"Unsupported gamma value {gamma} for ions. Available: {gammas}"
            )
        coeffs = COEFFS_I[gamma]
        out = {name: _thresh_i(beta_para, *params) for name, params in coeffs.items()}

    elif specie == "e":
        if gamma not in COEFFS_E:
            gammas = list(COEFFS_E.keys())
            raise ValueError(
                f"Unsupported gamma value {gamma} for electrons. Available: {gammas}"
            )
        coeffs = COEFFS_E[gamma]
        out = {name: _thresh_e(beta_para, *params) for name, params in coeffs.items()}

    else:
        raise ValueError(f"Unknown specie '{specie}'. Expected 'i' or 'e'.")

    return out
