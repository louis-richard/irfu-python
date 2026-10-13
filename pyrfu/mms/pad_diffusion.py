#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from ._psd_units import _output
from ._vdf_bins import _is_skymap, _pitch_angles

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"


def pad_diffusion(pad, d_aa):
    r"""Rate of change of a pitch-angle distribution by pitch-angle
    diffusion, as irfu-matlab PDist.pitchangle_diffusion.

    Parameters
    ----------
    pad : xarray.Dataset
        Pitch-angle distribution (time, energy, pitch angle), e.g. from
        `get_pitch_angle_dist`, with the pitch angles in degrees.
    d_aa : float or array_like
        Pitch-angle diffusion coefficient (rad^2/s): a scalar, or values
        broadcast against the data: (pitch angle,), (energy, pitch angle) or
        (time, energy, pitch angle).

    Returns
    -------
    out : xarray.Dataset
        Rate of change df/dt = 1 / sin(a) d/da (D_aa sin(a) df/da), in the
        units of `pad` per second.

    Raises
    ------
    ValueError
        If `pad` is not a pitch-angle distribution.

    Notes
    -----
    The derivatives are those of irfu-matlab: central differences inside,
    one-sided first-order differences at the first and last pitch angles
    (`numpy.gradient` with ``edge_order=1``, which also handles non-uniform
    pitch angles). The pitch angle is with respect to the magnetic field,
    i.e. far from the resonant energies, as in irfu-matlab. irfu-matlab
    differentiates in degrees and handles one time: its result is that of
    this function divided by (180 / pi)^2, for a single time.

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint, 1)
    >>> b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint, 1)
    >>> pad_e = mms.get_pitch_angle_dist(vdf_e, b_dmpa, angles=12)
    >>> df_dt = mms.pad_diffusion(pad_e, 1e-2)

    """
    if _is_skymap(pad):
        raise ValueError("pad must be a pitch-angle distribution")

    alpha = np.deg2rad(_pitch_angles(pad))
    data = pad.data.data
    d_aa = np.broadcast_to(np.asarray(d_aa, dtype=np.float64), data.shape)

    df_da = np.gradient(data, alpha, axis=2, edge_order=1)
    flux = d_aa * np.sin(alpha) * df_da
    df_dt = np.gradient(flux, alpha, axis=2, edge_order=1) / np.sin(alpha)

    units = pad.data.attrs.get("UNITS", "")
    return _output(pad, df_dt, f"{units}/s" if units else "1/s")
