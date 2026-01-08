#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
from typing import Optional, Union

# 3rd party imports
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

# Local imports
from pyrfu.mms.dsl2gse import _transformation_matrix
from pyrfu.pyrf.calc_fs import calc_fs
from pyrfu.pyrf.cotrans import cotrans
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.ts_tensor_xyz import ts_tensor_xyz
from pyrfu.pyrf.ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)

NDArrayFloats = NDArray[Union[np.float32, np.float64]]


def rotate_tensor(
    inp: DataArray,
    rot_flag: str,
    vec: Union[DataArray, NDArrayFloats, Dataset],
    perp: Optional[str] = "pp",
    verbose: Optional[bool] = False,
) -> DataArray:
    r"""Rotates pressure or temperature tensor into another coordinate system.

    Parameters
    ----------
    inp : DataArray
        Time series of the tensor to rotate. The tensor must be given in the
        form of a 3x3 matrix, where the first index is the time index and the
        last two indices are the tensor indices.

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

            * "gse" :   GSE coordinates, requires MMS spacecraft attitude DEFATT.

            * "gsm" :   GSM coordinates, requires MMS spacecraft attitude DEFATT.

    vec : DataArray or numpy.ndarray or xarray.Dataset
        Vector or coordinates system to rotate the tensor. If vec is time series of a
        vector tensor is rotated in field aligned coordinates. If vec is a
        numpy.ndarray rotates to a time independent coordinates system.

    perp : str, Optional
        Flag for perpendicular components of the tensor. Default is pp.
            * "pp" : perpendicular diagonal components are equal
            * "qq" : perpendicular diagonal components are most unequal

    verbose : bool, Optional
        Set to True to print additional information.

    Returns
    -------
    DataArray
        Time series of the pressure or temperature tensor in field-aligned,
        user-defined, or GSE coordinates.
        For "fac" Pe = [Ppar P12 P13; P12 Pperp1 P23; P13 P23 Pperp2].
        For "rot" and "gse" Pe = [Pxx Pxy Pxz; Pxy Pyy Pyz; Pxz Pyz Pzz]

    Raises
    ------
    TypeError
        * If inp is not a xarray.DataArray.
        * If vec is not a xarray.DataArray, numpy.ndarray or xarray.Dataset.
        * If rot_flag is not a string.
        * If perp is not a string.
    NotImplementedError
        * If rot_flag is not 'fac', 'rot', 'gse', or 'gsm'.
        * If rot_flag is 'rot' and vec is time dependent.

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

    """
    # Check input
    if not isinstance(inp, xr.DataArray):
        raise TypeError("inp must be a xarray.DataArray")

    if not isinstance(vec, (xr.DataArray, np.ndarray, xr.Dataset)):
        raise TypeError(
            "vec must be a xarray.DataArray, numpy.ndarray or xarray.Dataset"
        )

    if not isinstance(rot_flag, str):
        raise TypeError("rot_flag must be a string")

    if not isinstance(perp, str):
        raise TypeError("perp must be a string")

    # Get input data type
    precision = inp.data.dtype

    # Check input and load pressure/temperature terms
    inp_time: NDArray[np.datetime64] = inp.time.data
    inp_data: NDArrayFloats = inp.data
    n_t: int = len(inp_time)

    # Initialize rotation matrix
    rot_mat: NDArray[np.float64] = np.zeros((n_t, 3, 3), dtype=np.float64)

    if rot_flag.lower() == "fac":
        # If field-aligned coordinates vec must be the background magnetic field,
        # i.e., vector time series DataArray
        if not isinstance(vec, xr.DataArray):
            raise TypeError("vec must be a xarray.DataArray when rot_flag is 'fac'")

        # Resample magnetic field to the same time as the tensor
        b_back: DataArray = resample(vec, inp, f_s=calc_fs(inp))

        # Unit vector of the background magnetic field gives the parallel direction
        b_vec: NDArray[np.float64] = b_back.data.astype(np.float64)
        b_vec /= np.linalg.norm(b_vec, axis=1, keepdims=True)

        # Construct the rotation matrix
        # Parallel direction
        r_x: NDArray[np.float64] = b_vec
        # Perp1 arbitrarily chosen along x
        r_y: NDArray[np.float64] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        # Perp2 with correction
        r_z: NDArray[np.float64] = np.cross(r_x, r_y)
        r_z /= np.linalg.norm(r_z, axis=1, keepdims=True)
        r_y = np.cross(r_z, r_x)  # Corrected perp1 direction
        r_y /= np.linalg.norm(r_y, axis=1, keepdims=True)

        rot_mat[:, 0, :], rot_mat[:, 1, :], rot_mat[:, 2, :] = [r_x, r_y, r_z]

    elif rot_flag.lower() == "rot":
        # If arbitrary coordinates vec must be the new coordinate system, i.e.,
        # vector or a matrix. Currently only implemented for time independent
        # coordinate system.
        if not isinstance(vec, np.ndarray):
            raise TypeError("vec must be a numpy.ndarray when rot_flag is 'rot'")

        if vec.ndim == 1 and vec.shape[0] == 3:
            # First direction along the input vector
            r_x = vec.astype(np.float64)
            r_x /= np.linalg.norm(r_x, keepdims=True)
            # Second direction arbitrarily chosen along y
            r_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            # Third direction orthogonal to x and y directions
            r_z = np.cross(r_x, r_y)
            r_z /= np.linalg.norm(r_z, keepdims=True)
            # Corrected y direction
            r_y = np.cross(r_z, r_x)
            r_y /= np.linalg.norm(r_y, keepdims=True)

        elif vec.ndim == 2 and vec.shape[0] == 3 and vec.shape[1] == 3:
            vec = vec.astype(np.float64)
            # Make sure the rotation matrix is normalized and right-handed.
            r_x = vec[:, 0] / np.linalg.norm(vec[:, 0], keepdims=True)
            r_y = vec[:, 1] / np.linalg.norm(vec[:, 1], keepdims=True)

            r_z = np.cross(r_x, r_y) / np.linalg.norm(np.cross(r_x, r_y))
            r_y = np.cross(r_z, r_x) / np.linalg.norm(np.cross(r_z, r_x))

            # Check if the coordinate system has been changed and issue a warning
            if (
                abs(np.rad2deg(np.arccos(np.dot(r_y, vec[:, 0])))) > 1.0
                or abs(np.rad2deg(np.arccos(np.dot(r_z, vec[:, 2])))) > 1.0
            ):
                logging.warning(
                    "The new coordinate system has been changed to be right handed "
                    "orthogonal.",
                )

        else:
            raise NotImplementedError("Time dependent rotation matrix not implemented.")

        rot_mat[:, 0, :] = np.ones((n_t, 1), dtype=np.float64) * r_x
        rot_mat[:, 1, :] = np.ones((n_t, 1), dtype=np.float64) * r_y
        rot_mat[:, 2, :] = np.ones((n_t, 1), dtype=np.float64) * r_z

    elif rot_flag.lower() in ["gse", "gsm"]:
        # If GSE or GSM coordinates vec must be the spacecraft attitude, i.e.,
        # Dataset with z_ra and z_dec variables
        if not isinstance(vec, xr.Dataset):
            raise TypeError(
                "vec must be a xarray.Dataset when rot_flag is 'gse' or 'gsm'"
            )

        z_ra: NDArray[np.float64] = np.deg2rad(vec.z_ra.data.astype(np.float64))
        z_dec: NDArray[np.float64] = np.deg2rad(vec.z_dec.data.astype(np.float64))

        # Compute the spin axis direction in Geocentric equatorial inertial (GEI)
        # coordinates
        x: NDArray[np.float64] = np.cos(np.deg2rad(z_dec)) * np.cos(np.deg2rad(z_ra))
        y: NDArray[np.float64] = np.cos(np.deg2rad(z_dec)) * np.sin(np.deg2rad(z_ra))
        z: NDArray[np.float64] = np.sin(np.deg2rad(z_dec))
        sax_gei: DataArray = ts_vec_xyz(
            vec.time.data, np.transpose(np.vstack([x, y, z]))
        )

        # Compute the spin axis direction in the new coordinate system
        sax_rot: DataArray = cotrans(sax_gei, f"gei>{rot_flag.lower()}")
        spin_ax_rot: DataArray = resample(sax_rot, inp, f_s=calc_fs(inp))

        # Compute the rotation matrix
        rot_mat = _transformation_matrix(spin_ax_rot.data, 1)

    else:
        raise NotImplementedError(f"{rot_flag} method is not implemented.")

    p_tensor_p: NDArrayFloats = np.zeros((n_t, 3, 3), dtype=precision)

    for i in range(n_t):
        rot_temp: NDArrayFloats = np.squeeze(rot_mat[i, :, :].astype(precision))

        p_tensor_p[i, :, :] = np.matmul(
            np.matmul(rot_temp, np.squeeze(inp_data[i, :, :])),
            np.transpose(rot_temp),
        )

    if perp.lower() == "" or rot_flag.lower() in ["gse", "gsm"]:
        # maybe also add "rot" here??
        logging.info("No additional rotation applied.")
    elif perp.lower() == "pp":
        if verbose:
            logging.info(
                "Applying additional rotation to make the perpendicular components "
                "most equal"
            )
        thetas: NDArrayFloats = 0.5 * np.arctan(
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

    elif perp.lower() == "qq":

        if verbose:
            logging.info(
                "Applying additional rotation to make the perpendicular components "
                "most unequal"
            )

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
    else:
        raise NotImplementedError(f"{perp} method is not implemented.")

    # Construct output
    p_new: DataArray = ts_tensor_xyz(inp_time, p_tensor_p, attrs=inp.attrs)

    return p_new
