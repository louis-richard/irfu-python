#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Local imports
from ._psd_units import _output
from ._vdf_bins import _d3v, _is_skymap
from .vdf_convert import vdf_convert
from .vdf_velocity import vdf_velocity

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"


def vdf_partial_density(
    vdf,
    component: str = None,
    sc_pot=None,
    rotation=None,
    relativistic: bool = None,
):
    r"""Partial density or particle flux of each bin of a distribution, as
    irfu-matlab PDist.dn and PDist.dj.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution (time, energy, phi, theta) or, for the density
        only, pitch-angle distribution (time, energy, pitch angle), in any
        units of `vdf_convert`.
    component : {None, "x", "y", "z"}, Optional
        None for the partial density dn = f d3v (cm^-3), else the component of
        the partial particle flux dj_i = f v_i d3v (cm^-2 s^-1). Default is
        None.
    sc_pot : xarray.DataArray or float or array_like, Optional
        Spacecraft potential (V), see `vdf_velocity`. Default is no
        correction.
    rotation : array_like, Optional
        Rotation of the velocities for the particle flux, see
        `vdf_velocity`.
    relativistic : bool, Optional
        Relativistic phase-space density and speeds. Default is the
        ``relativistic`` attribute of `vdf` if any (True for the FEEPS
        skymaps), else False.

    Returns
    -------
    out : xarray.Dataset
        Copy of `vdf` with the partial densities or fluxes of the bins. Their
        sum over the energies and angles is the density (particle flux) of
        `psd_moments` with the same energy and angular bins.

    Raises
    ------
    ValueError
        If `component` is not None, "x", "y" or "z", or is given for a
        pitch-angle distribution.

    See Also
    --------
    pyrfu.mms.vdf_d3v, pyrfu.mms.vdf_velocity, pyrfu.mms.psd_moments

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 1)
    >>> dn_i = mms.vdf_partial_density(vdf_i)
    >>> n_i = dn_i.data.sum(dim=["idx0", "idx1", "idx2"])

    """
    if component not in [None, "x", "y", "z"]:
        raise ValueError(f"Invalid component {component}: None, x, y or z")

    if component is not None and not _is_skymap(vdf):
        raise ValueError("The particle flux needs a skymap")

    if relativistic is None:
        relativistic = vdf.attrs.get("relativistic", False)

    f_si = vdf_convert(vdf, "s^3/m^6", relativistic).data.data
    d_n = f_si * _d3v(vdf, sc_pot, relativistic)  # m^-3

    if component is None:
        return _output(vdf, d_n / 1e6, "cm^-3")

    velocity = vdf_velocity(vdf, sc_pot, rotation, relativistic)
    d_j = d_n * velocity[f"v{component}"].data * 1e3  # m^-2 s^-1
    return _output(vdf, d_j / 1e4, "1/(cm^2 s)")
