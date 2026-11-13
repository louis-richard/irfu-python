#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import constants

from pyrfu.mms._internal.epd import _check_flux
from pyrfu.mms._internal.psd_units import _MASS_RATIOS

# Local imports
from ..pyrf.resample import resample
from ..pyrf.ts_scalar import ts_scalar


def eis_moments(
    inp,
    specie: str = "proton",
    n_bg: xr.DataArray | None = None,
    p_bg: xr.DataArray | None = None,
):
    r"""Computes the partial moments given the omni-directional differential
    particle flux and the ion specie under the assumption of angular isotropy,
    with the relativistic speeds and momenta (as [1]_ for non-relativistic
    ions, which it gives below 1 MeV to 0.1 %)

    .. math::

        n = 4 \pi \sum_k \frac{J_k}{v_k} \Delta E_k

        \langle |v| \rangle = \frac{4 \pi}{n} \sum_k J_k \Delta E_k

        P = \frac{4 \pi}{3} \sum_k J_k p_k \Delta E_k

        T = \frac{P}{n k_B}

    where :math:`J_k` is the differential particle flux, :math:`E_k` the energy,
    :math:`v_k` and :math:`p_k` the relativistic speed and momentum and
    :math:`\Delta E_k` the width of the channel k. With the non-relativistic
    :math:`v_k = \sqrt{2 E_k / m}` and :math:`p_k = \sqrt{2 m E_k}` (it was
    non-relativistic), :math:`n = 4 \pi \sqrt{m / 2} \sum_k J_k E_k^{-1/2}
    \Delta E_k` and :math:`P = 2 / 3` of the energy density.

    Parameters
    ----------
    inp : xarray.DataArray
        Omni-directional differential particle flux, with the energy widths of
        the channels in its attributes (``delta_energy_minus`` and
        ``delta_energy_plus``, as :func:`pyrfu.mms.eis_omni` gives them).
    specie : {"proton", "helium", "alpha", "oxygen"}, Optional
        Particle specie ("alpha" is "helium", He2+; "oxygen" is O+). Default
        is "proton".
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
        If the specie is not supported, if `inp` has no energy widths, or if
        its ``UNITS`` are not those of a flux (counts and cps were accepted).

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
    0.93 to 1.01 now. The outputs have their units in ``UNITS`` (they had
    none).

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

    if specie not in ["proton", "helium", "alpha", "oxygen"]:
        raise ValueError(
            f"Invalid specie {specie}. Must be 'proton', 'helium', 'alpha' or "
            "'oxygen'"
        )

    # Masses of the PSD/flux conversions, 3.9726 (He2+) and 15.879 (O+) proton
    # masses (they were 4 and 16)
    mass = constants.proton_mass * _MASS_RATIOS[specie]

    _check_flux(inp)

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

    # Relativistic speeds and momenta of the channels (they were
    # non-relativistic), without cancellation: eps = E / (m c^2)
    eps = energy / (mass * constants.c**2)
    speed = constants.c * np.sqrt(eps * (2.0 + eps)) / (1.0 + eps)
    momentum = mass * constants.c * np.sqrt(eps * (2.0 + eps))

    # Sums over the channels, each over its width (Simpson's rule from the
    # first to the last channel centre left out the half channels at both
    # ends): the particles of a channel are 4 pi J / v dE
    def _sum(weight):
        return 4 * np.pi * np.sum(intensity * weight * delta_energy, axis=1)

    # Zeroth order moment number density m^-3
    n_i = _sum(1.0 / speed)

    # First order moment: mean speed <|v|> in m s^-1 (not the bulk velocity)
    v_i = _sum(1.0) / n_i

    # Second order moment: the pressure of an isotropic distribution, the
    # momentum flux p v / 3, in N m^-2 (Pa)
    p_i = _sum(momentum) / 3.0
    t_i = p_i / (n_i * constants.Boltzmann)  # K

    # Convert to usual units
    n_i = ts_scalar(inp.time.data, n_i * 1e-6, {"UNITS": "cm^-3"})
    v_i = ts_scalar(inp.time.data, v_i * 1e-3, {"UNITS": "km/s"})  # mean speed
    p_i = ts_scalar(inp.time.data, p_i * 1e9, {"UNITS": "nPa"})
    t_i = ts_scalar(
        inp.time.data,
        t_i * constants.Boltzmann / constants.elementary_charge,
        {"UNITS": "eV"},
    )

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
