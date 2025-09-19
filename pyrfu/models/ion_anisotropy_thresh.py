#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

EMPIRICAL_FITS = {
    "10^-2": {
        "proton-cyclotron": {"a": 0.69, "b": 0.4, "beta0": 0.0},
        "mirror": {"a": 1.040, "b": 0.633, "beta0": -0.012},
        "parallel-firehose": {"a": -0.647, "b": 0.583, "beta0": 0.713},
        "oblique-firehose": {"a": -1.447, "b": 1, "beta0": -0.148},
    }
}


def ion_anisotropy_thresh(beta, instability_type, growth_rate="10^-2"):
    r"""Compute the threshold using the empirical model defined in [1]_.

    .. math::

        R_i = T_{i\perp} / T_{i\parallel} = 1 + a / (\beta_{i\parallel} -\beta_0)^b


    Parameters
    ----------
    beta : xarray.DataArray or numpy.ndarray
        Time series or array of parallel ion plasma beta.
    a : float
        Coefficient from fit.
    b : float
        Coefficient from fit.
    beta0 : float
        Coefficient from fit.


    Returns
    -------
    r_i_thresh : xarray.DataArray
        Time series or array of threshold temperature anisotropy at the given beta.


    References
    ----------
    .. [1]  Hellinger, P., P. Travnicek, J. C. Kasper, and A. J. Lazarus (2006),
            Solar wind proton temperature anisotropy: Linear theory and WIND/SWE
            observations, Geophys. Res. Lett., 33, L09101, doi:10.1029/2006GL025925.

    """

    if growth_rate not in EMPIRICAL_FITS:
        raise ValueError(f"Growth rate {growth_rate} not recognized.")

    if instability_type not in EMPIRICAL_FITS[growth_rate]:
        raise ValueError(f"Instability type {instability_type} not recognized.")

    # Get fit parameters
    fit_params = EMPIRICAL_FITS[growth_rate][instability_type]
    a = fit_params["a"]
    b = fit_params["b"]
    beta0 = fit_params["beta0"]

    # Discard negative values from the denominator
    beta[beta < beta0] = np.nan

    # Compute threshold
    r_i_thresh = 1 + a / (beta - beta0) ** b

    return r_i_thresh
