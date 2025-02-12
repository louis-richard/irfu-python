#!/usr/bin/env python
# -*- coding: utf-8 -*-


from typing import Tuple

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.ts_scalar import ts_scalar
from pyrfu.pyrf.ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def dec_par_perp(
    inp: DataArray, b_bgd: DataArray, flag_spin_plane: bool = False
) -> Tuple[DataArray, DataArray, DataArray]:
    r"""Decomposes a vector into par/perp to B components. If flagspinplane
    decomposes components to the projection of ``b0`` into the XY plane.
    ``alpha`` gives the angle between ``b0`` and the XY. plane.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the field to decompose.
    b_bgd : xarray.DataArray
        Time series of the background magnetic field.
    flag_spin_plane : bool, Optional
        Flag if True gives the projection in XY plane.

    Returns
    -------
    a_para : xarray.DataArray
        Time series of the input field parallel to the background magnetic
        field.
    a_perp : xarray.DataArray
        Time series of the input field perpendicular to the background
        magnetic field.
    alpha : xarray.DataArray
        Time series of the angle between the background magnetic field and
        the XY plane.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field (FGM) and electric field (EDP)

    >>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_brst_l2", tint, mms_id)

    Decompose e_xyz into parallel and perpendicular to b_xyz components

    >>> e_para, e_perp, _ = pyrf.dec_par_perp(e_xyz, b_xyz)

    """

    # Check arguments types
    assert isinstance(inp, xr.DataArray), "inp must be an xarray.DataArray"
    assert isinstance(b_bgd, xr.DataArray), "b_bgd must be an xarray.DataArray"
    assert isinstance(flag_spin_plane, bool), "flag_spin_plane must be boolean"

    # Check inp and b_bgd shapes
    assert inp.ndim == 2 and inp.shape[1], "inp must be a vector"
    assert b_bgd.ndim == 2 and b_bgd.shape[1], "b_bgd must be a vector"

    # Resample background magnetic field to input
    b_bgd = resample(b_bgd, inp)

    # Get times and data
    inp_time = inp.time.data
    inp_data = inp.data
    b_bgd_data = b_bgd.data

    if not flag_spin_plane:
        b_mag = np.linalg.norm(b_bgd_data, axis=1, keepdims=True)

        indices = np.where(b_mag < 1e-3)[0]

        if indices.size > 0:
            b_mag[indices] = 1e-3 * np.ones((len(indices), 1))

        b_hat = b_bgd_data / b_mag

        a_para_arr = np.sum(b_hat * inp_data, axis=-1)
        a_para = ts_scalar(inp_time, a_para_arr)

        a_perp_arr = inp_data - (
            b_hat * np.tile(a_para_arr[:, np.newaxis], (1, inp_data.shape[1]))
        )
        a_perp = ts_vec_xyz(inp_time, a_perp_arr)

        alpha_arr = np.zeros_like(a_para_arr)
        alpha = ts_scalar(inp_time, alpha_arr)
    else:
        b_tot = np.sqrt(b_bgd_data[:, 0] ** 2 + b_bgd_data[:, 1] ** 2)
        b_hat = b_bgd_data / b_tot[:, np.newaxis]

        a_para_arr = inp_data[:, 0] * b_hat[:, 0] + inp_data[:, 1] * b_hat[:, 1]
        a_para = ts_scalar(inp_time, a_para_arr)

        a_perp_arr = inp_data[:, 0] * b_hat[:, 1] - inp_data[:, 1] * b_hat[:, 0]
        a_perp = ts_scalar(inp_time, a_perp_arr)

        alpha_arr = np.arctan2(b_bgd_data[:, 2], b_tot)
        alpha = ts_scalar(inp_time, alpha_arr)

    return a_para, a_perp, alpha
