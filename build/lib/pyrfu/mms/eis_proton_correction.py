#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _phxtof_calibration(energy, alpha, beta, gamma):
    r"""Pulse Height x Time Of Flight correction model from EPD Data Product
    Guide"""
    return 1 / (.5 * (1 + alpha * (np.tanh((energy - beta) / gamma) + 1)))


def _extof_calibration(energy, alpha, beta, gamma):
    r"""Energy x Time Of Flight correction model from EPD Data Product Guide"""
    return 1 / (.5 * (1 + alpha * (1 - np.tanh((energy - beta) / gamma) + 1)))


def eis_proton_correction(flux_eis):
    r"""Corrects proton flux values based on FPI/HPCA/EPD-EIS cross
    calibration. Correction to the EIS PHxTOF data are made by applying an
    energy-dependent numerical correction of the form:

    .. math:

        E_{PHxTOF} = \frac{1}{0.5*\left [1 + \alpha_{PH} \left (
        \operatorname{tanh}\left ( \frac{E - \beta_{PH}}{\gamma_{PH}} + 1
        \right ) \right ) \right ]}


    where E is energy and  :math:`\alpha_{PH} = -0.3` , :math:`\beta_{PH} =
    0.049` , :math:`\gamma_{PH} = 0.001` are coefficients. Minor adjustments
    were also made to the lowest energy EIS ExTOF data to correct for foil
    efficiencies. This correction is of the form:

    .. math:

        E_{ExTOF} = \frac{1}{0.5*\left [1 + \alpha_{E} \left (1 -
        \operatorname{tanh}\left ( \frac{E - \beta_{E}}{\gamma_{E}} + 1
        \right ) \right ) \right ]}

    where E is energy and :math:`\alpha_{E} = -0.3` , :math:`\beta_{E} =
    0.049` , :math:`\gamma_{E} = 0.001` are coefficients.

    Parameters
    ----------
    flux_eis : xarray.DataArray
        Omni-directional energy spectrum from EPD-EIS.

    Returns
    -------
    flux_eis_corr : xarray.DataArray
        Cross-calibrated omni-directional energy spectrum from EIS-EPD.

    See Also
    --------
    pyrfu.mms.get_eis_allt, pyrfu.mms.eis_omni

    """

    #  Coefficients from EPD Data Product Guide
    alpha_, beta_, gamma_ = [-.3, 49e-3, 1e-3]

    # Pulse Height x Time Of Flight (PHxTOF) energy correction factor
    energy_phxtof = flux_eis.energy.data[:7]
    phxtof_corr = _phxtof_calibration(energy_phxtof, alpha_, beta_, gamma_)

    # Energy x Time Of Flight (ExTOF) energy correction factor
    energy_extof = flux_eis.energy.data[7:]
    extof_corr = _extof_calibration(energy_extof, alpha_, beta_, gamma_)

    eis_corr = np.hstack([phxtof_corr, extof_corr])

    if isinstance(flux_eis, xr.Dataset):
        scopes_eis = list(filter(lambda x: x[0] == "t", flux_eis))
        out_keys = list(filter(lambda x: x not in scopes_eis, flux_eis))
        out_dict = {k: flux_eis[k] for k in out_keys}

        for scope in scopes_eis:
            out_dict[scope] = flux_eis[scope].copy()
            out_dict[scope].data *= eis_corr

        flux_eis_corr = xr.Dataset(out_dict)
    elif isinstance(flux_eis, xr.DataArray):
        # Apply correction to omni-directional energy spectrum
        flux_eis_corr = flux_eis.copy()
        flux_eis_corr.data *= eis_corr
    else:
        raise TypeError("flux_eis must be a Dataset or a DataArray")

    return flux_eis_corr
