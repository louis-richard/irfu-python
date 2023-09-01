#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf.calc_fs import calc_fs
from ..pyrf.resample import resample
from ..pyrf.ts_tensor_xyz import ts_tensor_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def rotate_tensor(inp, rot_flag, vec, perp: str = "pp"):
    """Rotates pressure or temperature tensor into another coordinate system.

    Parameters
    ----------
    PeIJ/Peall : xarray.DataArray
        Time series of either separated terms of the tensor or the complete
        tensor. If columns (PeXX,PeXY,PeXZ,PeYY,PeYZ,PeZZ)

    rot_flag : str
        Flag of the target coordinates system :
            * "fac" : 	Field-aligned coordinates, requires background
                        magnetic field Bback, optional flag "pp"
                        :math:`\\mathbf{P}_{\\perp 1} = \\mathbf{P}_{\\perp2}`
                        or "qq" :math:`\\mathbf{P}_{\\perp 1}` and
                        :math:`\\mathbf{P}_{\\perp 2}` are most unequal, sets
                        P_{23} to zero.

            * "rot" :	Arbitrary coordinate system, requires new x-direction
                        xnew, new y and z directions ynew, znew (if not
                        included y and z directions are orthogonal to xnew
                        and closest to the original y and z directions)

            * "gse" : GSE coordinates, requires MMS spacecraft number

    vec : xarray.DataArray or numpy.ndarray
        Vector or coordinates system to rotate the tensor. If vec is timeseries of a
        vector tensor is rotated in field aligned coordinates. If vec is a
        numpy.ndarray rotates to a time independant coordinates system.

    perp : str, Optional
        Flag for perpandicular components of the tensor. Default is pp.
            * "pp" : perpendicular diagonal components are equal
            * "qq" : perpendicular diagonal components are most unequal


    Returns
    -------
    Pe : xarray.DataArray
        Time series of the pressure or temperature tensor in field-aligned,
        user-defined, or GSE coordinates.
        For "fac" Pe = [Ppar P12 P13; P12 Pperp1 P23; P13 P23 Pperp2].
        For "rot" and "gse" Pe = [Pxx Pxy Pxz; Pxy Pyy Pyz; Pxz Pyz Pzz]

    Examples
    --------
    >>> from pyrfu import mms, pyrf
    >>> # Time interval
    >>> tint = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]
    >>> # Spacecraft index
    >>> mms_id = 1
    >>> # Load magnetic field and ion temperature tensor
    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)
    >>> t_xyz_i = mms.get_data("Ti_gse_fpi_fast_l2", tint, mms_id)
    >>> # Compute ion temperature in field aligned coordinates
    >>> t_xyzfac_i = mms.rotate_tensor(t_xyz_i, "fac", b_xyz, "pp")

    TODO : implement method "gse"
    """

    assert isinstance(rot_flag, str), "flag must be a string"
    assert rot_flag.lower() in ["fac", "rot", "gse"], "flag must be fac, rot or gse"

    assert isinstance(perp, str), "perp must be a string"
    assert perp.lower() in ["pp", "qq"], "perp must be pp or qq"

    # Check input and load pressure/temperature terms
    inp_times = inp.time.data
    n_t = len(inp_times)

    rot_mat = np.zeros((n_t, 3, 3))

    if rot_flag[0] == "f":
        assert isinstance(vec, xr.DataArray)
        b_back = resample(vec, inp, f_s=calc_fs(inp))

        b_vec = b_back / np.linalg.norm(b_back, axis=1, keepdims=True)

        r_x = b_vec.data
        r_y = np.array([1, 0, 0])
        r_z = np.cross(r_x, r_y)
        r_z /= np.linalg.norm(r_z, axis=1, keepdims=True)
        r_y = np.cross(r_z, r_x)
        r_y /= np.linalg.norm(r_y, axis=1, keepdims=True)

        rot_mat[:, 0, :], rot_mat[:, 1, :], rot_mat[:, 2, :] = [r_x, r_y, r_z]

    elif rot_flag[0] == "r":
        assert isinstance(vec, np.ndarray)

        if vec.ndim == 1 and vec.shape[0] == 3:
            r_x = vec
            r_x /= np.linalg.norm(r_x, keepdims=True)
            r_y = np.array([0, 1, 0])
            r_z = np.cross(r_x, r_y)
            r_z /= np.linalg.norm(r_z, keepdims=True)
            r_y = np.cross(r_z, r_x)
            r_y /= np.linalg.norm(r_y, keepdims=True)

        elif vec.ndim == 2 and vec.shape[0] == 3 and vec.shape[1] == 3:
            r_x = vec[:, 0] / np.linalg.norm(vec[:, 0], keepdims=True)
            r_y = vec[:, 1] / np.linalg.norm(vec[:, 1], keepdims=True)
            r_z = vec[:, 2] / np.linalg.norm(vec[:, 2], keepdims=True)

        else:
            raise TypeError("Vector format not recognized.")

        rot_mat[:, 0, :] = np.ones((n_t, 1)) * r_x
        rot_mat[:, 1, :] = np.ones((n_t, 1)) * r_y
        rot_mat[:, 2, :] = np.ones((n_t, 1)) * r_z

    else:
        raise NotImplementedError("gse is not yet implemented!!")

    p_tensor_p = np.zeros((n_t, 3, 3))

    for i in range(n_t):
        rot_temp = np.squeeze(rot_mat[i, :, :])

        p_tensor_p[i, :, :] = np.matmul(
            np.matmul(rot_temp, np.squeeze(inp.data[i, :, :])),
            np.transpose(rot_temp),
        )

    if perp.lower() == "pp":
        thetas = 0.5 * np.arctan(
            (p_tensor_p[:, 2, 2] - p_tensor_p[:, 1, 1]) / (2 * p_tensor_p[:, 1, 2]),
        )
        thetas[np.isnan(thetas)] = 0.0

        for i, theta in enumerate(thetas):
            rot_temp = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(theta), np.sin(theta)],
                    [0, -np.sin(theta), np.cos(theta)],
                ],
            )

            p_tensor_p[i, :, :] = np.matmul(
                np.matmul(rot_temp, np.squeeze(p_tensor_p[i, :, :])),
                np.transpose(rot_temp),
            )

    else:
        thetas = 0.5 * np.arctan(
            (2 * p_tensor_p[:, 1, 2]) / (p_tensor_p[:, 2, 2] - p_tensor_p[:, 1, 1]),
        )

        for i, theta in enumerate(thetas):
            rot_temp = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)],
                ],
            )

            p_tensor_p[i, :, :] = np.matmul(
                np.matmul(rot_temp, np.squeeze(p_tensor_p[i, :, :])),
                np.transpose(rot_temp),
            )

    # Construct output
    p_new = ts_tensor_xyz(inp_times, p_tensor_p, attrs=inp.attrs)

    return p_new
