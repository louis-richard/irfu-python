#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Local imports
from ..pyrf.anisotropy_thresholds import anisotropy_thresholds

# Growth rates in units of the proton cyclotron frequency
GROWTH_RATES = {"10^-2": 0.01, "10^-3": 0.001, "10^-4": 0.0001}

# Names of the instabilities in pyrf.anisotropy_thresholds
INSTABILITY_TYPES = {
    "proton-cyclotron": "proton cyclotron",
    "mirror": "mirror mode",
    "parallel-firehose": "parallel firehose",
    "oblique-firehose": "oblique firehose",
}


def ion_anisotropy_thresh(beta, instability_type, growth_rate="10^-2"):
    r"""Compute the threshold using the empirical model defined in [1]_ and
    the fit parameters of [2]_.

    .. math::

        R_i = T_{i\perp} / T_{i\parallel} = 1 + a / (\beta_{i\parallel} -\beta_0)^b

    This is a wrapper of :func:`pyrfu.pyrf.anisotropy_thresholds`.

    Parameters
    ----------
    beta : float or numpy.ndarray or xarray.DataArray
        Time series or array of parallel ion plasma beta. It is not modified.
    instability_type : {"proton-cyclotron", "mirror", "parallel-firehose",
                        "oblique-firehose"}
        Instability.
    growth_rate : {"10^-2", "10^-3", "10^-4"}, Optional
        Maximum growth rate of the instability in units of the proton cyclotron
        frequency. Default is "10^-2".

    Returns
    -------
    r_i_thresh : float or numpy.ndarray or xarray.DataArray
        Time series or array of threshold temperature anisotropy at the given
        beta, of the same type as `beta`. NaN where the fit is undefined
        (:math:`\beta_{i\parallel} \leq \beta_0`).

    Raises
    ------
    ValueError
        If `growth_rate` or `instability_type` is not recognized.

    References
    ----------
    .. [1]  Hellinger, P., P. Travnicek, J. C. Kasper, and A. J. Lazarus (2006),
            Solar wind proton temperature anisotropy: Linear theory and WIND/SWE
            observations, Geophys. Res. Lett., 33, L09101, doi:10.1029/2006GL025925.

    .. [2]  Verscharen, D., B. D. G. Chandran, K. G. Klein, and E. Quataert
            (2016), Collisionless isotropization of the solar-wind protons by
            compressive fluctuations and plasma instabilities, Astrophys. J.,
            831, 128, doi:10.3847/0004-637X/831/2/128.

    """

    if growth_rate not in GROWTH_RATES:
        raise ValueError(f"Growth rate {growth_rate} not recognized.")

    if instability_type not in INSTABILITY_TYPES:
        raise ValueError(f"Instability type {instability_type} not recognized.")

    thresholds = anisotropy_thresholds(beta, "i", GROWTH_RATES[growth_rate])
    r_i_thresh = thresholds[INSTABILITY_TYPES[instability_type]]

    return r_i_thresh
