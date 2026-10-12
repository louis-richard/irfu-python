#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from ._psd_units import _output
from ._vdf_bins import _band_solid_angles, _broadcast_angles, _is_skymap

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"


def vdf_solid_angle(vdf):
    r"""Solid angle of each bin of a skymap or a pitch-angle distribution, as
    irfu-matlab PDist.solidangle.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution (time, energy, phi, theta) or pitch-angle
        distribution (time, energy, pitch angle), e.g. from
        `get_pitch_angle_dist`, with the angles in degrees.

    Returns
    -------
    out : xarray.Dataset
        Copy of `vdf` whose data are the solid angles (sr) of the bins.

    Raises
    ------
    ValueError
        If `vdf` is neither a skymap nor a pitch-angle distribution.

    Notes
    -----
    The bins span the angular widths of the attributes (``delta_phi_*`` and
    ``delta_theta_*``, ``delta_pitchangle_*``) when they exist, else half the
    spacing of the angles on each side, as irfu-matlab. The solid angle of a
    skymap bin is d_phi (cos(theta_1) - cos(theta_2)); a pitch-angle bin is a
    band of 2 pi (cos(alpha_1) - cos(alpha_2)). The bins of a full grid sum to
    4 pi.

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint, 1)
    >>> d_omega = mms.vdf_solid_angle(vdf_e)

    """
    _is_skymap(vdf)
    data = vdf.data.data
    solid_angle = _broadcast_angles(_band_solid_angles(vdf), data)
    return _output(vdf, np.broadcast_to(solid_angle, data.shape).copy(), "sr")
