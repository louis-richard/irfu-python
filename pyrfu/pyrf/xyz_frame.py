#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .resample import resample
from .ts_tensor_xyz import ts_tensor_xyz


def _vectors(inp, ref):
    r"""Vectors (time, 3) of a time series resampled to the times `ref`, or
    of a 3-vector repeated."""

    if isinstance(inp, xr.DataArray):
        if inp.ndim != 2 or inp.shape[1] != 3:
            raise ValueError("x and y must be time series of vectors or 3-vectors")

        return resample(inp, ref).data.astype(np.float64)

    inp = np.asarray(inp, dtype=np.float64)

    if inp.shape != (3,):
        raise ValueError("x and y must be time series of vectors or 3-vectors")

    return np.tile(inp, (len(ref), 1))


def xyz_frame(x, y, time=None):
    r"""Frame (x, y, z) defined by two directions: x along `x`, y along the
    part of `y` perpendicular to `x`, and z = x cross y.

    Parameters
    ----------
    x : xarray.DataArray or array_like
        Time series of the vectors along x, or a 3-vector, e.g. the magnetic
        field.
    y : xarray.DataArray or array_like
        Time series of the vectors in the (x, y) plane, or a 3-vector, in the
        coordinate system of `x`, e.g. E x B or the bulk velocity.
    time : xarray.DataArray or xarray.Dataset, Optional
        Time series (e.g. a distribution) whose times the frame is at.
        Default is the times of `x`, or of `y` if `x` is a 3-vector.

    Returns
    -------
    out : xarray.DataArray
        Time series (time, 3, 3) of the unit vectors x, y and z as columns,
        as `pyrfu.mms.reduce` and `pyrfu.pyrf.new_xyz` take them, NaN where
        `y` is along `x` (no plane), with the attribute ``xyz`` "columns" by
        which `pyrfu.mms.vdf_shift`, `pyrfu.mms.vdf_rebin`,
        `pyrfu.mms.vdf_velocity` and `pyrfu.mms.vdf_projection` recognise
        it (they take the unit vectors as rows otherwise).

    Raises
    ------
    ValueError
        If `x` or `y` is neither a time series of vectors nor a 3-vector, or
        if neither is a time series and `time` is not given.

    Notes
    -----
    The time series are resampled to the times of the frame. Common frames:
    ``xyz_frame(b_xyz, pyrf.cross(e_xyz, b_xyz))`` (x along B, y along
    E x B), ``xyz_frame(b_xyz, v_xyz)`` (x along B, y along the bulk velocity
    perpendicular to B, z along B x V) and ``xyz_frame(b_xyz, [0, 0, 1])``
    (x along B, y along the spin axis perpendicular to B, in DMPA).

    See Also
    --------
    pyrfu.mms.reduce, pyrfu.pyrf.new_xyz, pyrfu.pyrf.convert_fac

    Examples
    --------
    >>> from pyrfu import mms, pyrf
    >>> tint = ["2015-10-30T05:15:42.000", "2015-10-30T05:15:54.000"]
    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint, 1)
    >>> b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint, 1)
    >>> e_dsl = mms.get_data("e_dsl_edp_brst_l2", tint, 1)

    Electron distributions reduced to the plane (B, E x B)

    >>> xyz = pyrf.xyz_frame(b_dmpa, pyrf.cross(e_dsl, b_dmpa), time=vdf_e)
    >>> f2d = mms.reduce(vdf_e, xyz, "2d", "cart")

    """

    if time is None:
        time = next((v for v in [x, y] if isinstance(v, xr.DataArray)), None)

    if time is None:
        raise ValueError("x or y must be a time series, or give time")

    ref = time.time
    x_vec, y_vec = [_vectors(v, ref) for v in [x, y]]

    # x along x, y the part of y perpendicular to x, z = x cross y; NaN
    # without a plane
    with np.errstate(divide="ignore", invalid="ignore"):
        x_hat = x_vec / np.linalg.norm(x_vec, axis=1, keepdims=True)
        y_perp = y_vec - np.sum(y_vec * x_hat, axis=1, keepdims=True) * x_hat
        norm = np.linalg.norm(y_perp, axis=1, keepdims=True)
        tiny = norm <= 1e-12 * np.linalg.norm(y_vec, axis=1, keepdims=True)
        y_hat = np.where(tiny, np.nan, y_perp / norm)

    z_hat = np.cross(x_hat, y_hat)

    out = ts_tensor_xyz(ref.data, np.stack([x_hat, y_hat, z_hat], axis=2))
    out.attrs["xyz"] = "columns"

    return out
