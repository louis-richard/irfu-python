#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Local imports
from ._psd_units import _output
from ._vdf_bins import (
    _band_solid_angles,
    _broadcast_angles,
    _broadcast_energy,
    _is_skymap,
    _speed_edges,
)
from .vdf_convert import vdf_convert

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"


def vdf_flux(vdf, sc_pot=None, per_sr: bool = True, relativistic: bool = None):
    r"""Particle flux of each bin of a skymap or a pitch-angle distribution,
    as irfu-matlab PDist.flux.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution (time, energy, phi, theta) or pitch-angle
        distribution (time, energy, pitch angle), in any units of
        `vdf_convert`.
    sc_pot : xarray.DataArray or float or array_like, Optional
        Spacecraft potential (V), see `vdf_velocity`. Default is no
        correction.
    per_sr : bool, Optional
        Flux per steradian, 1/(cm^2 s sr) (True, default), or flux through
        the solid angle of the bin, 1/(cm^2 s).
    relativistic : bool, Optional
        Relativistic phase-space density and speeds. Default is the
        ``relativistic`` attribute of `vdf` if any (True for the FEEPS
        skymaps), else False.

    Returns
    -------
    out : xarray.Dataset
        Copy of `vdf` with the flux of the particles of each bin, the integral
        of f v^3 dv over the energy channel, (v_2^4 - v_1^4) / 4 f (times the
        solid angle if `per_sr` is False).

    Notes
    -----
    The energy and angular bins are those of `vdf_d3v`. irfu-matlab
    PDist.flux uses the electron mass for all species.

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint, 1)
    >>> flux_e = mms.vdf_flux(vdf_e)

    """
    _is_skymap(vdf)

    if relativistic is None:
        relativistic = vdf.attrs.get("relativistic", False)

    f_si = vdf_convert(vdf, "s^3/m^6", relativistic).data.data
    v_lower, v_upper = _speed_edges(vdf, sc_pot, relativistic)
    flux = f_si * _broadcast_energy((v_upper**4 - v_lower**4) / 4, f_si) / 1e4

    if per_sr:
        return _output(vdf, flux, "1/(cm^2 s sr)")

    flux = flux * _broadcast_angles(_band_solid_angles(vdf), f_si)
    return _output(vdf, flux, "1/(cm^2 s)")
