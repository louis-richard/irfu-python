#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Optional, Union

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.pyrf.calc_fs import calc_fs
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def convert_fac(
    inp: DataArray,
    b_bgd: DataArray,
    r_xyz: Optional[Union[DataArray, np.ndarray, list]] = None,
) -> DataArray:
    r"""Transform to a field-aligned coordinate (FAC) system.

    The FAC system is defined as :
        * R_parallel_z aligned with the background magnetic field
        * R_perp_y defined by R_parallel cross the position vector of the
            spacecraft (nominally eastward at the equator)
        * R_perp_x defined by R_perp_y cross R_par

    If inp is one vector along r direction, out is inp[perp, para]
    projection.

    Parameters
    ----------
    inp : DataArray
        Time series of the input field.
    b_bgd : DataArray
        Time series of the background magnetic field.
    r_xyz : xarray.DataArray or ndarray or list
        Position vector of spacecraft.

    Returns
    -------
    DataArray
        Time series of the input field in field aligned coordinates
        system.

    Raises
    ------
    TypeError
        * If inp is not a xarray.DataArray.
        * If b_bgd is not a xarray.DataArray.
        * If r_xyz is not a xarray.DataArray or ndarray or list.
    ValueError
        * If inp is not a scalar or a vector.
        * If b_bgd is not a vector or a tensor.
        * If r_xyz is not a vector.

    Notes
    -----
    All input parameters must be in the same coordinate system.

    Examples
    --------
    >>> import numpy
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field (FGM) and electric field (EDP)

    >>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_brst_l2", tint, mms_id)

    Convert to field aligned coordinates

    >>> e_xyzfac = pyrf.convert_fac(e_xyz, b_xyz, numpy.array([1, 0, 0]))

    """

    # Check input type
    if not isinstance(inp, xr.DataArray):
        raise TypeError("inp must be a xarray.DataArray")

    if not isinstance(b_bgd, xr.DataArray):
        raise TypeError("b_xyz must be a xarray.DataArray")

    if r_xyz is None:
        r_xyz = np.array([1, 0, 0])
    elif not isinstance(r_xyz, (xr.DataArray, np.ndarray, list)):
        raise TypeError("r_xyz must be a xarray.DataArray or ndarray or list")

    if len(inp) != len(b_bgd):
        b_bgd = resample(b_bgd, inp, f_s=calc_fs(inp))

    time: np.ndarray = inp.time.data
    inp_data: np.ndarray = inp.data
    b_bgd_data: np.ndarray = b_bgd.data

    if isinstance(r_xyz, xr.DataArray) and r_xyz.ndim == 2 and r_xyz.shape[1] == 3:
        r_xyz_ts: DataArray = resample(r_xyz, b_bgd, f_s=calc_fs(b_bgd))
        r_xyz_data: np.ndarray = r_xyz_ts.data
    elif isinstance(r_xyz, (list, np.ndarray)) and len(r_xyz) == 3:
        r_xyz_data = np.tile(r_xyz, (len(b_bgd), 1))
    else:
        raise ValueError("r_xyz must be a vector (time series or time independent)")

    if b_bgd.ndim == 2 and b_bgd.shape[1] == 3:
        # Normalize background magnetic field
        b_hat: np.ndarray = b_bgd_data / np.linalg.norm(b_bgd, axis=1, keepdims=True)

        # Perpendicular
        r_perp_y: np.ndarray = np.cross(b_hat, r_xyz_data, axis=1)
        r_perp_y /= np.linalg.norm(r_perp_y, axis=1, keepdims=True)
        r_perp_x: np.ndarray = np.cross(r_perp_y, b_bgd, axis=1)
        r_perp_x /= np.linalg.norm(r_perp_x, axis=1, keepdims=True)
        r_para: np.ndarray = b_hat
    elif b_bgd.ndim == 3 and b_bgd.shape[1] == 3 and b_bgd.shape[2] == 3:
        r_perp_x = b_bgd_data[:, 0]
        r_perp_y = b_bgd_data[:, 1]
        r_para = b_bgd_data[:, 2]
    else:
        raise ValueError("b_bgd must be a vector or a tensor time series")

    if inp_data.ndim == 2 and inp_data.shape[1] == 3:
        out_data = np.zeros(inp.shape, dtype=inp_data.dtype)

        out_data[:, 0] = np.sum(r_perp_x * inp_data, axis=1)
        out_data[:, 1] = np.sum(r_perp_y * inp_data, axis=1)
        out_data[:, 2] = np.sum(r_para * inp_data, axis=1)

        # To xarray
        out = ts_vec_xyz(time, out_data, attrs=inp.attrs)

    elif inp_data.ndim == 1:
        out_data = np.zeros([inp_data.shape[0], 2])

        out_data[:, 0] = inp * np.sum(r_perp_x * r_xyz_data, axis=1)
        out_data[:, 1] = inp * np.sum(r_para * r_xyz_data, axis=1)

        out = xr.DataArray(
            out_data, coords=[time, ["perp", "para"]], dims=["time", "comp"]
        )
    else:
        raise ValueError(
            "inp must be a vector or scalar. See pyrfu.mms.rotate_tensor for tensor."
        )

    return out
