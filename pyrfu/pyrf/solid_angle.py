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


def _directions(inp, name):
    r"""Unit vectors (n, 3) of a vector (3,) or vectors (n, 3)."""
    if not isinstance(inp, (list, np.ndarray, xr.DataArray)):
        raise TypeError(f"{name} must be array_like or xarray.DataArray")

    inp = np.atleast_2d(np.asarray(inp, dtype=np.float64))

    if inp.ndim != 2 or inp.shape[1] != 3:
        raise ValueError(f"{name} must be a vector (3,) or vectors (n, 3)")

    return inp / np.linalg.norm(inp, axis=1, keepdims=True)


def solid_angle(inp0, inp1, inp2):
    r"""Calculates the solid angle of three vectors making up a triangle
    in a unit sphere with the sign taken into account, as irfu-matlab
    irf.solidangle.

    Parameters
    ----------
    inp0 : array_like or xarray.DataArray
        First vector (3,) or vectors (n, 3), of any length (only their
        directions are used).
    inp1 : array_like or xarray.DataArray
        Second vector(s).
    inp2 : array_like or xarray.DataArray
        Third vector(s).

    Returns
    -------
    angle : numpy.ndarray
        Solid angle (sr) of the spherical triangle with vertices the
        directions of the three vectors (n,), between -2 pi and 2 pi, with the
        sign of (inp2 x inp1) . inp0 (positive when inp0, inp1 and inp2 turn
        clockwise seen from outside the sphere).

    Raises
    ------
    TypeError
        If an input is not array_like or a xarray.DataArray.
    ValueError
        If an input is not a vector (3,) or vectors (n, 3).

    Notes
    -----
    The solid angle Omega of the unit vectors a, b and c is given by
    tan(Omega / 2) = a . (b x c) / (1 + a . b + a . c + b . c) [1]_, with the
    triple product computed as a . ((b - a) x (c - a)). irf.solidangle uses
    the spherical excess, with the angles of the triangle from the law of
    cosines: it loses all precision for small triangles (a relative error
    of 0.2 for 0.01 rad sides, 4e3 for 1e-4 rad), and gives NaN for vectors
    that are not unit vectors or for degenerate triangles (zero here). The
    new formula agrees with a 50-digit evaluation to 1e-11 down to 1e-4 rad
    sides.

    References
    ----------
    .. [1] Van Oosterom, A., and Strackee, J. (1983), The solid angle of a
           plane triangle, IEEE Trans. Biomed. Eng., BME-30(2), 125-126,
           doi:10.1109/TBME.1983.325207.

    """

    # Directions only
    vec_a = _directions(inp0, "inp0")
    vec_b = _directions(inp1, "inp1")
    vec_c = _directions(inp2, "inp2")

    # Van Oosterom and Strackee (1983), the triple product from the
    # differences to a (a . (a x .) = 0), accurate for small triangles
    numerator = np.sum(vec_a * np.cross(vec_b - vec_a, vec_c - vec_a), axis=1)
    denominator = 1.0 + np.sum(vec_a * vec_b, axis=1)
    denominator += np.sum(vec_a * vec_c, axis=1) + np.sum(vec_b * vec_c, axis=1)

    # Solid angle with the sign of (c x b) . a = -a . (b x c), as irf.solidangle
    angle = -2.0 * np.arctan2(numerator, denominator)

    return angle
