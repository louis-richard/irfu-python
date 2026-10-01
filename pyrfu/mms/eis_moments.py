#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import constants, integrate

# Local imports
from ..pyrf.resample import resample
from ..pyrf.ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def eis_moments(
    inp,
    specie: str = "proton",
    n_bg: xr.DataArray = None,
    p_bg: xr.DataArray = None,
):
    r"""Computes the partial moments given the omni-directional differential
    particle flux and the ion specie under the assumption of angular isotropy
    and non-relativistic ions using the formula from [1]_

    .. math::

        n = 4 \pi \sqrt{\frac{m_i}{2}} \int J E^{-1/2} \textrm{d}E

        \langle |v| \rangle = \frac{4 \pi}{n} \int J \textrm{d}E

        P = \frac{2}{3} 4 \pi \sqrt{\frac{m_i}{2}} \int J E^{1/2}
        \textrm{d}E

        T = \frac{P}{n k_B}

    where :math:`J` is the differential particle flux and :math:`E` the energy
    (the integral in :math:`P` is the energy density).

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
    v : xarray.DataArray
        Time series of the mean speed :math:`\langle |v| \rangle` in [km/s]
        (not the bulk velocity, which is zero for an isotropic distribution).
    p : xarray.DataArray
        Time series of the pressure in [nPa]
    t : xarray.DataArray
        Time series of the temperature in [eV]

    Raises
    ------
    ValueError
        If the specie is not supported.

    Notes
    -----
    The input omni-directional differential particle flux must be given in
    [1/(cm^2 s sr keV)], and the energy must be in [keV]. The integration is
    performed over the energy channels using the composite Simpson's rule, so
    these are partial moments, and an energy channel with NaN flux gives NaN
    moments at that time.

    References
    ----------
    .. [1]  Mauk, B. H., D. G. Mitchell, R. W. McEntire, C. P. Paranicas,
            E. C. Roelof, D. J. Williams, S. M. Krimigis, and A. Lagg (2004),
            Energetic ion characteristics and neutral gas interactions in
            Jupiter's magnetosphere, J. Geophys. Res., 109, A09S12,
            doi:10.1029/2003JA010270.

    """

    if specie not in ["proton", "alpha", "oxygen"]:
        raise ValueError(
            f"Invalid specie {specie}. Must be 'proton', 'alpha' or 'oxygen'"
        )

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

    fact = 4 * np.pi * np.sqrt(mass / 2)

    # Define the integrand as a function of the differential particle flux,
    # the energy and the moment order
    def _int(flux_c, energy_c, order):
        return np.sqrt(energy_c) ** order * (flux_c / energy_c) * np.sqrt(energy_c)

    # Zeroth order moment number density m^-3
    n_i = fact * integrate.simpson(_int(intensity, energy, 0), x=energy, axis=1)

    # First order moment: mean speed <|v|> in m s^-1 (not the bulk velocity)
    v_i = fact * integrate.simpson(_int(intensity, energy, 1), x=energy, axis=1)
    v_i /= n_i * np.sqrt(mass / 2)

    # Second order moment: the integral is the energy density u, and the pressure
    # of an isotropic distribution is P = 2 u / 3, in N m^-2 (Pa)
    u_i = fact * integrate.simpson(_int(intensity, energy, 2), x=energy, axis=1)
    p_i = 2.0 * u_i / 3.0
    t_i = p_i / (n_i * constants.Boltzmann)  # K

    # Convert to usual units
    n_i = ts_scalar(inp.time.data, n_i * 1e-6)  # cm^-3
    v_i = ts_scalar(inp.time.data, v_i * 1e-3)  # mean speed, km s^-1
    p_i = ts_scalar(inp.time.data, p_i * 1e9)  # nPa
    t_i = ts_scalar(
        inp.time.data,
        t_i * constants.Boltzmann / constants.elementary_charge,
    )  # eV

    if n_bg is not None and p_bg is not None:
        # Resample background to differential particle flux sampling
        n_bg = resample(n_bg, n_i)
        p_bg = resample(p_bg, p_i)

        # Remove background density
        n_i.data -= n_bg.data

        # Remove background pressure
        p_i.data -= 3.0 * p_bg
        p_i.data += constants.proton_mass * n_i * v_i**2 * (n_bg / (n_i - n_bg))

        # Update temperature
        t_i.data = 1e-9 * p_i.data / (1e6 * n_i.data * constants.electron_volt)

    return n_i, v_i, p_i, t_i
