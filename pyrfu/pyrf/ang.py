#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from xarray import DataArray

# Local imports
from .resample import resample
from .ts_scalar import ts_scalar


def ang(inp1, inp2):
    r"""Angle between two vectors, in degrees, as irfu-matlab ``irf_ang`` (for
    vectors in cartesian coordinates).

    .. math::

        \theta = \arctan\left(\frac{|\mathbf{X} \times \mathbf{Y}|}
        {\mathbf{X} \cdot \mathbf{Y}}\right)

    The arctangent of the ratio keeps the precision of the small angles and of
    the angles close to 180 degrees, where the arccosine of the normalised dot
    product (irfu-matlab) loses it.

    Parameters
    ----------
    inp1 : xarray.DataArray or array_like
        Time series of the first vector X, a vector (3 values) or an array of
        vectors (n, 3).
    inp2 : xarray.DataArray or array_like
        Time series of the second vector Y (resampled to the times of `inp1`
        if both are time series, as `dot`), a vector or an array of vectors.

    Returns
    -------
    out : xarray.DataArray or numpy.ndarray
        Angle between X and Y, from 0 to 180 degrees, NaN where X or Y is
        zero: a time series on the times of the time series input (of `inp1`
        if both are), else an array (a scalar for two vectors).

    Raises
    ------
    ValueError
        If `inp1` or `inp2` is not a vector or vectors of 3 components.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Magnetic field and ion bulk velocity

    >>> b_xyz = mms.get_data("b_gse_fgm_srvy_l2", tint, 1)
    >>> v_xyz_i = mms.get_data("vi_gse_fpi_fast_l2", tint, 1)

    Angle between the ion velocity and the magnetic field

    >>> theta_vb = pyrf.ang(v_xyz_i, b_xyz)

    """

    # The angle is symmetric: the time series first
    if isinstance(inp2, DataArray) and not isinstance(inp1, DataArray):
        inp1, inp2 = inp2, inp1

    if isinstance(inp1, DataArray) and isinstance(inp2, DataArray):
        inp2 = resample(inp2, inp1)

    x_xyz = np.asarray(getattr(inp1, "data", inp1), dtype=np.float64)
    y_xyz = np.asarray(getattr(inp2, "data", inp2), dtype=np.float64)

    if x_xyz.shape[-1:] != (3,) or y_xyz.shape[-1:] != (3,):
        raise ValueError("inp1 and inp2 must be vectors of 3 components")

    sin_xy = np.linalg.norm(np.cross(x_xyz, y_xyz), axis=-1)
    cos_xy = np.sum(x_xyz * y_xyz, axis=-1)
    theta = np.rad2deg(np.arctan2(sin_xy, cos_xy))

    # No direction for a zero vector
    zero = np.linalg.norm(x_xyz, axis=-1) == 0
    zero |= np.linalg.norm(y_xyz, axis=-1) == 0
    theta = np.where(zero, np.nan, theta)

    if isinstance(inp1, DataArray):
        return ts_scalar(inp1.time.data, theta, {"UNITS": "deg"})

    return theta[()]
