#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

from scipy import constants, integrate

# Local imports
from ..pyrf import ts_scalar, resample

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def eis_moments(inp, specie: str = "proton", n_bg: xr.DataArray = None,
                p_bg: xr.DataArray = None):
    r"""Computes the partial moments given the omni-directional differential
    particle flux and the ion specie under the assumption of angular isotropy
    and non-relativistic ions using the formula from [1]_

    .. math::

        n \\left [ m^{-3} \\right ] = 4 \\pi \\sqrt{\\frac{m_i}{2}} \\sum_{i}
        \\left ( E_i^{1/2} \\right)^0 \\left ( \\frac{J_i}{E_i}\\right)
        \\left ( E_i^{1/2} \\textrm{d} E_i\\right)

        P \\left [ Pa \\right ] = 4 \\pi \\sqrt{\\frac{m_i}{2}} \\sum_{i}
        \\left ( E_i^{1/2} \\right)^2 \\left ( \\frac{J_i}{E_i}\\right)
        \\left ( E_i^{1/2} \\textrm{d} E_i\\right)

        T \\left [ K \\right ] = \\frac{P}{n k_b}

    Parameters
    ----------
    inp : xarray.DataArray
        Omni-directional differential particle flux.
    specie : {"proton", "alpha", "oxygen"}, Optional
        Particle specie. Default is "proton".
    n_bg : xarray.DataArray, Optional
        Time series of the background density. If None do not remove
        penetrating radiations.
    p_bg : xarray.DataArray, Optional
        Time series of the background pressure. If None do not remove the
        penetrating radiations.

    Returns
    -------
    n : xarray.DataArray
        Time series of the number density in [cm^{-3}]
    p : xarray.DataArray
        Time series of the pressure in [nPa]
    t : xarray.DataArray
        Time series of the temperature in [eV]

    Notes
    -----
    The input omni-directional differential particle flux must be given in
    [(1/cm^2 s sr keV)^{-1}], and the energy must be in [keV].
    The integration is performed using the composite Simpson’s rule.

    References
    ----------
    .. [1]  Mauk, B. H., D. G. Mitchell, R. W. McEntire, C. P. Paranicas,
            E. C. Roelof, D. J. Williams, S. M. Krimigis, and A. Lagg (2004),
            Energetic ion characteristics and neutral gas interactions in
            Jupiter’s magnetosphere, J. Geophys. Res., 109, A09S12,
            doi:10.1029/2003JA010270.

    """

    assert specie in ["proton", "alpha", "oxygen"]

    if specie == "proton":
        mass = constants.proton_mass
    elif specie == "alpha":
        mass = constants.proton_mass * 4
    else:
        mass = constants.proton_mass * 16

    # Convert energy and differential particle flux to SI units keV -> J and
    # 1/(cm^2 s sr keV) -> 1/(m^2 s sr J)
    energy = 1e3 * constants.electron_volt * inp.energy.data.copy()
    intensity = 1e4 * inp.data.copy() / (1e3 * constants.electron_volt)

    a_ = 4 * np.pi * np.sqrt(mass / 2)

    # Define the integrand as a function of the differential particle flux,
    # the energy and the moment order
    def _int(i_, e_, n_):
        return np.sqrt(e_) ** n_ * (i_ / e_) * np.sqrt(e_)

    # Zeroth order moment number density m^-3
    n = a_ * integrate.simps(_int(intensity, energy, 0), energy, axis=1)

    # First order moment bulk velocity m s^-1
    v = a_ * integrate.simps(_int(intensity, energy, 1), energy, axis=1)
    v /= n * np.sqrt(mass / 2)
    # Second order moment pressure in N m^-2 (Pa)
    p = a_ * integrate.simps(_int(intensity, energy, 2), energy, axis=1)
    t = p / (n * constants.Boltzmann)  # K

    # Convert to usual units
    n = ts_scalar(inp.time.data, n * 1e-6)  # cm^-3
    v = ts_scalar(inp.time.data, v * 1e-3)  # km s^-1
    p = ts_scalar(inp.time.data, p * 1e9)  # nPa
    t = ts_scalar(inp.time.data,
                  t * constants.Boltzmann / constants.elementary_charge)  # eV

    if n_bg is not None and p_bg is not None:
        # Resample background to differential particle flux sampling
        n_bg = resample(n_bg, n)
        p_bg = resample(p_bg, p)

        # Remove background density
        n.data -= n_bg.data

        # Remove background pressure
        p.data -= 3. * p_bg
        p.data += constants.proton_mass * n * v ** 2 * (n_bg / (n - n_bg))

        # Update temperature
        t.data = 1e-9 * p.data / (1e6 * n.data * constants.electron_volt)

    return n, v, p, t
