#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def solid_angle(inp0, inp1, inp2):
    r"""Calculates the solid angle of three vectors making up a triangle
    in a unit sphere with the sign taken into account.

    Parameters
    ----------
    inp0 : ndarray
        First vector.
    inp1 : ndarray
        Second vector.
    inp2 : ndarray
        Third vector.

    Returns
    -------
    angle : float
        Solid angle.

    """

    message = "must be array_like of xarray.DataArray"
    assert isinstance(inp0, (list, np.ndarray, xr.DataArray)), f"inp0 {message}"
    assert isinstance(inp1, (list, np.ndarray, xr.DataArray)), f"inp2 {message}"
    assert isinstance(inp2, (list, np.ndarray, xr.DataArray)), f"inp2 {message}"

    inp0, inp1, inp2 = [np.atleast_2d(inp) for inp in [inp0, inp1, inp2]]

    message = "must be a vector"
    assert inp0.shape[1] == 3, f"inp0 {message}"
    assert inp1.shape[1] == 3, f"inp1 {message}"
    assert inp2.shape[1] == 3, f"inp2 {message}"

    # Calculate the smaller angles between the vectors around origin
    acos_12 = np.arccos(np.sum(inp2 * inp1, axis=1))
    acos_02 = np.arccos(np.sum(inp0 * inp2, axis=1))
    acos_01 = np.arccos(np.sum(inp1 * inp0, axis=1))

    # Calculate the angles in the spherical triangle (Law of Cosines)
    alpha = np.arccos(
        (np.cos(acos_12) - np.cos(acos_02) * np.cos(acos_01))
        / (np.sin(acos_02) * np.sin(acos_01)),
    )
    beta = np.arccos(
        (np.cos(acos_02) - np.cos(acos_12) * np.cos(acos_01))
        / (np.sin(acos_12) * np.sin(acos_01)),
    )
    gamma = np.arccos(
        (np.cos(acos_01) - np.cos(acos_02) * np.cos(acos_12))
        / (np.sin(acos_02) * np.sin(acos_12)),
    )

    # Calculates the Surface area on the unit sphere (solid angle)
    angle = alpha + beta + gamma - np.pi
    # Calculate the sign of the area
    var = np.cross(inp2, inp1)
    div = np.sum(var * inp0, axis=1)
    sgn = np.sign(div)

    # Solid angle with sign taken into account
    angle = sgn * angle

    return angle
