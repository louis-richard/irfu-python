#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .resample import resample
from .ts_scalar import ts_scalar
from .ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _check_method(flag_method):
    default_value = 0
    if flag_method.lower() == "e_perp+nan":
        default_value = np.nan

        flag_method = "e.b=0"

    assert flag_method.lower() in ["e.b=0", "e_par"], "Invalid flag"

    return flag_method, default_value


def edb(e_xyz, b_bgd, angle_lim: float = 20.0, flag_method: str = "E.B=0"):
    r"""Compute Ez under assumption :math:`\mathbf{E}.\mathbf{B}=0` or
    :math:`\mathbf{E}.\mathbf{B} \approx 0`

    Parameters
    ----------
    e_xyz : xarray.DataArray
        Time series of the electric field, in the spin-plane coordinates
        (e.g., DSL): Ex and Ey are used, Ez is replaced.
    b_bgd : xarray.DataArray
        Time series of the background magnetic field.
    angle_lim : float, Optional
        Limit on the elevation of B above the spin plane, in degrees. With
        "E.B=0" and "E_perp+NaN", Ez is computed where the elevation is more
        than angle_lim (Bz large enough to divide by), and set to 0 (NaN with
        "E_perp+NaN") elsewhere. With "E_par", Ez is computed where the
        elevation is less than angle_lim, and set to 0 elsewhere. Default is
        20.
    flag_method : {"E.B=0", "E_par", "E_perp+NaN"}, Optional
        Assumption on the measured electric field (case-insensitive):

        * "E.B=0": E perpendicular to B,
          :math:`E_z = -(E_x B_x + E_y B_y) / B_z`. Default.
        * "E_par": the field measured along the projection of B on the spin
          plane is a parallel field,
          :math:`E_z = (E_x B_x + E_y B_y) B_z / (B_x^2 + B_y^2)`.
        * "E_perp+NaN": as "E.B=0", with Ez set to NaN where it is not
          computed.

    Returns
    -------
    ed : xarray.DataArray
        Time series of the electric field with the computed Ez.
    d : xarray.DataArray
        Time series of the elevation of B above the spin plane, in degrees.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]

    Spacecraft indices

    >>> mms_id = 1

    Load magnetic field, electric field and spacecraft position

    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_fast_l2", tint, mms_id)

    Compute Ez

    >>> e_z, alpha = pyrf.edb(e_xyz, b_xyz)

    """

    assert isinstance(e_xyz, xr.DataArray), "e_xyz must be a xarray.DataArray"
    assert isinstance(b_bgd, xr.DataArray), "b_bgd must be a xarray.DataArray"
    assert isinstance(angle_lim, (int, float)), "angle_lime must be int or float"

    flag_method, default_value = _check_method(flag_method)

    # Make sure that background magnetic field sampling matches the
    # electric field sampling
    b_bgd = resample(b_bgd, e_xyz)

    b_data = b_bgd.data
    e_data = e_xyz.data.copy()  # Ez is overwritten: don't change the caller's E
    e_data[:, -1] *= default_value

    if flag_method.lower() == "e.b=0":
        # Calculate using assumption E.B=0
        b_angle = np.arctan2(
            b_data[:, 2],
            np.linalg.norm(b_data[:, :2], axis=1),
        )
        b_angle = np.rad2deg(b_angle)
        ind = np.abs(b_angle) > angle_lim

        if True in ind:
            e_data[ind, 2] = -np.sum(
                e_data[ind, :2] * b_data[ind, :2],
                axis=1,
            )
            e_data[ind, 2] /= b_data[ind, 2]

    else:
        # Calculate using assumption that E field along the B projection is
        # coming from parallel electric field
        b_angle = np.arctan2(
            b_data[:, 2],
            np.linalg.norm(b_data[:, :2], axis=1),
        )
        b_angle = np.rad2deg(b_angle)
        ind = np.abs(b_angle) < angle_lim

        if True in ind:
            e_data[ind, 2] = np.sum(e_data[ind, :2] * b_data[ind, :2], axis=1)
            e_data[ind, 2] *= b_data[ind, 2]
            e_data[ind, 2] /= np.linalg.norm(b_data[ind, :2], axis=1) ** 2

    b_angle = ts_scalar(e_xyz.time.data, b_angle, {"UNITS": "degrees"})
    e_data = ts_vec_xyz(
        e_xyz.time.data,
        e_data,
        dict(e_xyz.attrs),
    )

    return e_data, b_angle
