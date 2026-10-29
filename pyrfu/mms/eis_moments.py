#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import constants

# Local imports
from ..pyrf.resample import resample
from ..pyrf.ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
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

        n = 4 \pi \sqrt{\frac{m_i}{2}} \sum_k J_k E_k^{-1/2} \Delta E_k

        \langle |v| \rangle = \frac{4 \pi}{n} \sum_k J_k \Delta E_k

        P = \frac{2}{3} 4 \pi \sqrt{\frac{m_i}{2}} \sum_k J_k E_k^{1/2}
        \Delta E_k

        T = \frac{P}{n k_B}

    where :math:`J_k` is the differential particle flux, :math:`E_k` the energy
    and :math:`\Delta E_k` the width of the channel k (the sum in :math:`P` is
    the energy density).

    Parameters
    ----------
    inp : xarray.DataArray
        Omni-directional differential particle flux, with the energy widths of
        the channels in its attributes (``delta_energy_minus`` and
        ``delta_energy_plus``, as :func:`pyrfu.mms.eis_omni` gives them).
    specie : {"proton", "helium", "alpha", "oxygen"}, Optional
        Particle specie ("alpha" is "helium"). Default is "proton".
    n_bg : xarray.DataArray, Optional
        Time series of the background density of EIS in cm^-3, to subtract.
        If None (default), no background is removed.
    p_bg : xarray.DataArray, Optional
        Time series of the background pressure of EIS in nPa, to subtract.
        If None (default), no background is removed.

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
        If the specie is not supported, or if `inp` has no energy widths.

    Notes
    -----
    The input omni-directional differential particle flux must be given in
    [1/(cm^2 s sr keV)], and the energy and its widths must be in [keV]. The
    moments are sums over the energy channels, each over its width: they are
    partial moments, and an energy channel with NaN flux gives NaN moments at
    that time. They were computed with Simpson's rule from the first to the
    last channel centre, which left out the half channels at both ends: for
    an isotropic proton Maxwellian on the ExTOF channels, n was 0.39 (kT = 10
    keV) to 0.94 (100 keV) of the partial density over the channels, against
    0.93 to 1.01 now.

    With `n_bg` and `p_bg`, the background density and pressure are
    subtracted, and the temperature computed from the difference; the mean
    speed is not corrected. The pressure was reduced by 3 `p_bg`, as the trace
    of the pressure tensor in the correction of the FPI moments
    (:func:`pyrfu.mms.remove_imoms_background`). They must be the background
    of EIS, in the same partial moments: not the FPI-DIS background
    (``nbgi``, ``pbgi``), the penetrating radiation in FPI below 30 keV. The EIS
    files have no background product; `n_bg` and `p_bg` will be deprecated.

    References
    ----------
    .. [1]  Mauk, B. H., D. G. Mitchell, R. W. McEntire, C. P. Paranicas,
            E. C. Roelof, D. J. Williams, S. M. Krimigis, and A. Lagg (2004),
            Energetic ion characteristics and neutral gas interactions in
            Jupiter's magnetosphere, J. Geophys. Res., 109, A09S12,
            doi:10.1029/2003JA010270.

    """

    masses = {"proton": 1, "helium": 4, "alpha": 4, "oxygen": 16}

    if specie not in masses:
        raise ValueError(
            f"Invalid specie {specie}. Must be 'proton', 'helium', 'alpha' or "
            "'oxygen'"
        )

    mass = constants.proton_mass * masses[specie]

    if not {"delta_energy_minus", "delta_energy_plus"} <= set(inp.attrs):
        raise ValueError(
            "inp must have the energy widths of the channels in its attributes "
            "(delta_energy_minus, delta_energy_plus), as eis_omni gives them"
        )

    # Convert energy, channel widths and differential particle flux to SI units
    # keV -> J and 1/(cm^2 s sr keV) -> 1/(m^2 s sr J)
    energy = 1e3 * constants.electron_volt * inp.energy.data.copy()
    delta_energy = (
        1e3
        * constants.electron_volt
        * (
            np.asarray(inp.attrs["delta_energy_minus"])
            + np.asarray(inp.attrs["delta_energy_plus"])
        )
    )
    intensity = 1e4 * inp.data.copy() / (1e3 * constants.electron_volt)

    fact = 4 * np.pi * np.sqrt(mass / 2)

    # Sum over the channels, each over its width (Simpson's rule from the first
    # to the last channel centre left out the half channels at both ends)
    def _sum(order):
        integrand = energy ** (order / 2) * (intensity / energy) * np.sqrt(energy)
        return np.sum(integrand * delta_energy, axis=1)

    # Zeroth order moment number density m^-3
    n_i = fact * _sum(0)

    # First order moment: mean speed <|v|> in m s^-1 (not the bulk velocity)
    v_i = fact * _sum(1)
    v_i /= n_i * np.sqrt(mass / 2)

    # Second order moment: the sum is the energy density u, and the pressure of
    # an isotropic distribution is P = 2 u / 3, in N m^-2 (Pa)
    u_i = fact * _sum(2)
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

        # Remove the background density and pressure (the pressure was reduced
        # by 3 p_bg, plus a term in m_p n v^2 of about 1e-27 nPa)
        n_i.data -= n_bg.data
        p_i.data -= p_bg.data

        # Update temperature (eV)
        t_i.data = 1e-9 * p_i.data / (1e6 * n_i.data * constants.electron_volt)

    return n_i, v_i, p_i, t_i
