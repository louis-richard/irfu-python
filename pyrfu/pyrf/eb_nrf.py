#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .resample import resample
from .ts_vec_xyz import ts_vec_xyz


def eb_nrf(e_xyz, b_xyz, v_xyz, flag="a"):
    """Find E and B in MP system given B and MP normal vector.

    Parameters
    ----------
    e_xyz : xarray.DataArray
        Time series of the electric field.
    b_xyz : xarray.DataArray
        Time series of the magnetic field.
    v_xyz : xarray.DataArray or array_like
        Time series of the normal vector, or constant normal vector [vx, vy, vz].
    flag : str or array_like, Optional
        Method flag. Default is "a".
        * a : L is along b_xyz, N closest to v_xyz and M = NxL
        * b : N is along v_xyz, L is the mean direction of b_xyz in plane perpendicular
        to N (without the NaN samples of b_xyz), and M = NxL
        * array_like : N is along v_xyz , L is closest to the direction specified by
        L_vector (e.g., maximum variance direction), M = NxL

    Returns
    -------
    out : xarray.DataArray
        Time series of the electric field in the (L, M, N) coordinates, in this
        order, at the times of e_xyz, with the attributes of e_xyz and
        ``COORDINATE_SYSTEM`` set to "lmn".

    Raises
    ------
    TypeError
        If e_xyz or b_xyz is not a xarray.DataArray, or flag not a str or
        array_like.
    ValueError
        If e_xyz, b_xyz, v_xyz or an array_like flag is not a vector, or a
        str flag not "a" or "b".

    """
    # Check inputs
    for name, inp in [("e_xyz", e_xyz), ("b_xyz", b_xyz)]:
        if not isinstance(inp, xr.DataArray):
            raise TypeError(f"{name} must be a xarray.DataArray")

        if inp.ndim != 2 or inp.shape[1] != 3:
            raise ValueError(f"{name} must be a vector")

    if isinstance(v_xyz, xr.DataArray):
        if v_xyz.ndim != 2 or v_xyz.shape[1] != 3:
            raise ValueError("v_xyz must be a vector")

        if not np.array_equal(v_xyz.time.data, e_xyz.time.data):
            v_xyz = resample(v_xyz, e_xyz)

        v_data = v_xyz.data
    else:
        v_data = np.array(v_xyz, dtype=float)
        if v_data.shape != (3,):
            raise ValueError("constant v_xyz must be a 3 components vector")
        v_data = np.tile(v_data, (len(e_xyz), 1))

    if not isinstance(flag, (str, np.ndarray, list, tuple)):
        raise TypeError("flag must be a str or array_like")

    if isinstance(flag, str):
        if flag.lower() not in ["a", "b"]:
            raise ValueError(f'flag must be "a" or "b", not {flag!r}')

        flag_case = flag.lower()
        l_direction = None

    else:
        flag = np.array(flag)
        if flag.ndim != 1 or len(flag) != 3:
            raise ValueError("array_like flag must be a vector of 3 components")
        l_direction = flag
        flag_case = "c"

    if flag_case == "a":
        b_xyz = resample(b_xyz, e_xyz)

        n_l = b_xyz.data / np.linalg.norm(b_xyz.data, axis=1, keepdims=True)
        n_n = np.cross(np.cross(b_xyz.data, v_data), b_xyz.data)
        n_n /= np.linalg.norm(n_n, axis=1, keepdims=True)
        n_m = np.cross(n_n, n_l)  # in (vn x b) direction
    elif flag_case == "b":
        n_n = v_data / np.linalg.norm(v_data, axis=1, keepdims=True)
        # One NaN sample of B made L and M NaN everywhere
        n_m = np.cross(n_n, np.nanmean(b_xyz.data, axis=0))
        n_m /= np.linalg.norm(n_m, axis=1, keepdims=True)
        n_l = np.cross(n_m, n_n)

    else:
        n_n = v_data / np.linalg.norm(v_data, axis=1, keepdims=True)
        n_m = np.cross(n_n, l_direction)
        n_m /= np.linalg.norm(n_m, axis=1, keepdims=True)
        n_l = np.cross(n_m, n_n)

    # estimate e in new coordinates
    e_lmn = np.vstack([np.sum(e_xyz.data * vec, axis=1) for vec in [n_l, n_m, n_n]])

    # Attributes of e_xyz, in the new frame (as mva)
    attrs = {**e_xyz.attrs, "COORDINATE_SYSTEM": "lmn"}
    out = ts_vec_xyz(e_xyz.time.data, np.transpose(e_lmn), attrs)

    return out
