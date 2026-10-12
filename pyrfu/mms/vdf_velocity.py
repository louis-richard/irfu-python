#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ._vdf_bins import (
    _kinetic_energy,
    _mass,
    _relativistic,
    _sc_pot,
    _speed,
)

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"


def _rotation_matrices(rotation, n_t):
    # (time, 3, 3) rows are the new unit vectors
    if rotation is None:
        return np.tile(np.eye(3), (n_t, 1, 1))

    rotation = np.asarray(rotation, dtype=np.float64)

    if rotation.shape == (3, 3):
        return np.tile(rotation, (n_t, 1, 1))

    if rotation.shape == (n_t, 3, 3):
        return rotation

    raise ValueError("rotation must be a 3x3 matrix or (time, 3, 3) matrices")


def vdf_velocity(vdf, sc_pot=None, rotation=None, relativistic: bool = None):
    r"""Velocity of the particles of each bin of a skymap, as irfu-matlab
    PDist.v.

    Parameters
    ----------
    vdf : xarray.Dataset
        Skymap distribution (time, energy, phi, theta), with the look
        directions phi (time, phi) or (phi,) and theta (theta,) in degrees
        and the energies (time, energy) in eV.
    sc_pot : xarray.DataArray or float or array_like, Optional
        Spacecraft potential (V): a time series, resampled to the times of
        `vdf`, a scalar or (time,) values. The energies are corrected as in
        irfu-matlab PDist.v: ions gain and electrons lose e V. Default is no
        correction.
    rotation : array_like, Optional
        3x3 matrix, or (time, 3, 3) matrices, whose rows are the new unit
        vectors in the coordinate system of `vdf`, e.g. [e_perp1, e_perp2, b]
        for field-aligned coordinates. Default is no rotation.
    relativistic : bool, Optional
        Relativistic speeds. Default is the ``relativistic`` attribute of
        `vdf` if any (True for the FEEPS skymaps), else False.

    Returns
    -------
    out : xarray.Dataset
        Velocities ``vx``, ``vy`` and ``vz`` (time, energy, phi, theta) in
        km/s, at the energies of the bins. The direction of a bin is minus its
        look direction; ``out / numpy.linalg.norm`` gives the unit vectors of
        irfu-matlab PDist.xyz.

    Raises
    ------
    ValueError
        If `sc_pot` or `rotation` have the wrong shape.

    See Also
    --------
    pyrfu.mms.vdf_d3v, pyrfu.mms.vdf_partial_density

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_i = mms.get_data("pdi_fpi_brst_l2", tint, 1)
    >>> v_i = mms.vdf_velocity(vdf_i)

    """
    n_t = len(vdf.time)
    energy = np.broadcast_to(
        np.asarray(vdf.energy.data, dtype=np.float64), vdf.data.shape[:2]
    )
    energy = _kinetic_energy(vdf, energy, _sc_pot(vdf, sc_pot))
    speed = _speed(energy, _mass(vdf), _relativistic(vdf, relativistic)) / 1e3

    phi = np.deg2rad(np.broadcast_to(vdf.phi.data, (n_t, vdf.data.shape[2])))
    theta = np.deg2rad(np.asarray(vdf.theta.data, dtype=np.float64))
    phi, theta = phi[:, np.newaxis, :, np.newaxis], theta[np.newaxis, np.newaxis, :]

    # Particle velocity is minus the look direction
    v_xyz = [
        -speed[..., np.newaxis, np.newaxis] * np.sin(theta) * np.cos(phi),
        -speed[..., np.newaxis, np.newaxis] * np.sin(theta) * np.sin(phi),
        -speed[..., np.newaxis, np.newaxis] * np.cos(theta) * np.ones_like(phi),
    ]
    v_xyz = np.stack(np.broadcast_arrays(*v_xyz), axis=-1)
    v_xyz = np.einsum("tij,tepqj->tepqi", _rotation_matrices(rotation, n_t), v_xyz)

    attrs = {"UNITS": "km/s"}
    out = xr.Dataset(
        {
            name: vdf.data.copy(data=v_xyz[..., i])
            for i, name in enumerate(["vx", "vy", "vz"])
        }
    )

    for name in ["vx", "vy", "vz"]:
        out[name].attrs = dict(attrs)

    out.attrs = {"species": vdf.attrs.get("species")}
    return out
