#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from .resample import resample
from .ts_scalar import ts_scalar
from .ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _check_method(flag_method):
    default_value = 0
    if flag_method.lower() == "e_perp+nan":
        default_value = np.nan

        flag_method = "e.b=0"

    assert flag_method.lower() in ["e.b=0", "e_par"], "Invalid flag"

    return flag_method, default_value


def edb(e_xyz, b_bgd, angle_lim: float = 20., flag_method: str = "E.B=0"):
    r"""Compute Ez under assumption :math:`\mathbf{E}.\mathbf{B}=0` or
    :math:`\mathbf{E}.\mathbf{B} \approx 0`

    Parameters
    ----------
    e_xyz : xarray.DataArray
        Time series of the electric field.
    b_bgd : xarray.DataArray
        Time series of the background magnetic field.
    angle_lim : float, Optional
        B angle with respect to the spin plane should be less than angle_lim
        degrees otherwise Ez is set to 0. Default is 20.
    flag_method : str, Optional
        Assumption on the direction of the measured electric field :
            "e.b=0" :  :math:`\mathbf{E}.\mathbf{B}=0`. (Default)
            "e_par" :  :math:`\mathbf{E}` field along the B projection is
            coming from parallelelectric field.
            "e_perp+nan" : to fill.

    Returns
    -------
    ed : xarray.DataArray
        Time series of the electric field output.
    d : xarray.DataArray
        Time series of the B elevation angle above spin plane.

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

    flag_method, default_value = _check_method(flag_method)

    # Make sure that background magnetic field sampling matches the
    # electric field sampling
    b_bgd = resample(b_bgd, e_xyz)

    b_data = b_bgd.data
    e_data = e_xyz.data
    e_data[:, -1] *= default_value

    if flag_method.lower() == "e.b=0":
        # Calculate using assumption E.B=0
        b_angle = np.arctan2(b_data[:, 2],
                             np.linalg.norm(b_data[:, :2], axis=1))
        b_angle = np.rad2deg(b_angle)
        ind = np.abs(b_angle) > angle_lim

        if True in ind:
            e_data[ind, 2] = -np.sum(e_data[ind, :2] * b_data[ind, :2], axis=1)
            e_data[ind, 2] /= b_data[ind, 2]

    else:
        # Calculate using assumption that E field along the B projection is
        # coming from parallel electric field
        b_angle = np.arctan2(b_data[:, 2],
                             np.linalg.norm(b_data[:, :2], axis=1))
        b_angle = np.rad2deg(b_angle)
        ind = np.abs(b_angle) < angle_lim

        if True in ind:
            e_data[ind, 2] = np.sum(e_data[ind, :2] * b_data[ind, :2], axis=1)
            e_data[ind, 2] *= b_data[ind, 2]
            e_data[ind, 2] /= np.linalg.norm(b_data[:, :2], axis=1) ** 2

    b_angle = ts_scalar(e_xyz.time.data, b_angle, {"UNITS": "degrees"})
    e_data = ts_vec_xyz(e_xyz.time.data, e_data,
                        {"UNITS": e_xyz.attrs["UNITS"]})

    return e_data, b_angle
