#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .resample import resample
from .ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def eb_nrf(e_xyz, b_xyz, v_xyz, flag=0):
    """Find E and B in MP system given B and MP normal vector.

    Parameters
    ----------
    e_xyz : xarray.DataArray
        Time series of the electric field.
    b_xyz : xarray.DataArray
        Time series of the magnetic field.
    v_xyz : xarray.DataArray
        Normal vector.
    flag : str or ndarray
        Method flag :
        * a : L is along b_xyz, N closest to v_xyz and M = NxL
        * b : N is along v_xyz, L is the mean direction of b_xyz in plane perpendicular
        to N, and M = NxL
        * numpy,ndarray : N is along v_xyz , L is closest to the direction specified by
        L_vector (e.g., maximum variance direction), M = NxL

    Returns
    -------
    out : xarray.DataArray
        to fill.

    """
    # Check inputs
    assert isinstance(e_xyz, xr.DataArray), "e_xyz must be a xarray.DataArray"
    assert isinstance(b_xyz, xr.DataArray), "b_xyz must be a xarray.DataArray"
    assert isinstance(v_xyz, xr.DataArray), "v_xyz must be a xarray.DataArray"

    assert e_xyz.ndim == 2 and e_xyz.shape[1] == 3, "e_xyz must be a vector"
    assert b_xyz.ndim == 2 and b_xyz.shape[1] == 3, "e_xyz must be a vector"
    assert v_xyz.ndim == 2 and v_xyz.shape[1] == 3, "e_xyz must be a vector"

    assert isinstance(flag, (str, np.ndarray, list)), "Invalid flag type"

    if isinstance(flag, str):
        assert flag.lower() in ["a", "b"], "flag must be a or b"
        flag_case = flag
        l_direction = None

    else:
        flag = np.array(flag)
        assert flag.ndim == 1 and len(flag) == 3, "array_like flag must be a vector!"
        l_direction = flag
        flag_case = "c"

    if flag_case == "a":
        b_xyz = resample(b_xyz, e_xyz)

        n_l = b_xyz.data / np.linalg.norm(b_xyz.data, axis=1, keepdims=True)
        n_n = np.cross(np.cross(b_xyz.data, v_xyz.data), b_xyz.data)
        n_n /= np.linalg.norm(n_n, axis=1, keepdims=True)
        n_m = np.cross(n_n, n_l)  # in (vn x b) direction
    elif flag_case == "b":
        n_n = v_xyz.data / np.linalg.norm(v_xyz, axis=1, keepdims=True)
        n_m = np.cross(n_n, np.mean(b_xyz.data, axis=0))
        n_m /= np.linalg.norm(n_m, axis=1, keepdims=True)
        n_l = np.cross(n_m, n_n)

    else:
        n_n = v_xyz.data / np.linalg.norm(v_xyz, axis=1, keepdims=True)
        n_m = np.cross(n_n, l_direction)
        n_m /= np.linalg.norm(n_m, axis=1, keepdims=True)
        n_l = np.cross(n_m, n_n)

    # estimate e in new coordinates
    e_lmn = np.vstack([np.sum(e_xyz.data * vec, axis=1) for vec in [n_l, n_m, n_n]])
    out = ts_vec_xyz(e_xyz.time.data, np.transpose(e_lmn), e_xyz.attrs)

    return out
