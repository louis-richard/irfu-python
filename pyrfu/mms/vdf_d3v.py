#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Local imports
from ._psd_units import _data_and_units, _output
from ._vdf_bins import _d3v, _is_skymap

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"

# Volume units matching the phase-space density units, factor from m^3/s^3
_VOLUME_UNITS = {
    "s^3/cm^6": ("cm^3/s^3", 1e6),
    "s^3/km^6": ("km^3/s^3", 1e-9),
}


def vdf_d3v(vdf, sc_pot=None, relativistic: bool = None):
    r"""Velocity-space volume of each bin of a skymap or a pitch-angle
    distribution, as irfu-matlab PDist.d3v.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution (time, energy, phi, theta) or pitch-angle
        distribution (time, energy, pitch angle), with the energies in eV and
        the angles in degrees.
    sc_pot : xarray.DataArray or float or array_like, Optional
        Spacecraft potential (V), see `vdf_velocity`. Default is no
        correction.
    relativistic : bool, Optional
        Relativistic speeds. Default is the ``relativistic`` attribute of
        `vdf` if any (True for the FEEPS skymaps), else False.

    Returns
    -------
    out : xarray.Dataset
        Copy of `vdf` whose data are the volumes of the bins,
        (v_2^3 - v_1^3) / 3 times their solid angle, in cm^3/s^3 for
        phase-space densities in s^3/cm^6 (km^3/s^3 for s^3/km^6, else
        m^3/s^3), so that ``vdf * d3v`` is the partial density of each bin.

    Notes
    -----
    The energy edges of the channels are those of the ``delta_energy_*``
    attributes when they match the energies, else the midpoints between the
    channels; the angular bins are those of `vdf_solid_angle`. The spacecraft
    potential is applied with the sign of the charge, as in `vdf_velocity`
    and `psd_moments` (irfu-matlab PDist.d3v subtracts it for all species).

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 1)
    >>> d3v_i = mms.vdf_d3v(vdf_i)

    """
    _is_skymap(vdf)
    _, units = _data_and_units(vdf)
    new_units, factor = _VOLUME_UNITS.get(units.lower(), ("m^3/s^3", 1.0))
    return _output(vdf, _d3v(vdf, sc_pot, relativistic) * factor, new_units)
