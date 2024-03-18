#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Literal, Optional, Union

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

# Local imports
from ..pyrf.calc_fs import calc_fs
from ..pyrf.cotrans import cotrans
from ..pyrf.resample import resample
from ..pyrf.ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _transformation_matrix(
    spin_axis: np.ndarray, direction: Literal[-1, 1]
) -> np.ndarray:
    r"""Return transformation matrix from DSL to GSE.

    Parameters
    ----------
    spin_axis : numpy.ndarray
        Spin axis.
    direction : {-1, 1}
        Direction of the transformation.

    Returns
    -------
    numpy.ndarray
        Transformation matrix.

    Raises
    ------
    ValueError
        If direction is not -1 or 1.

    """
    r_x, r_y, r_z = [spin_axis[:, i] for i in range(3)]

    fact = 1.0 / np.sqrt(r_y**2 + r_z**2)
    out = np.zeros((len(fact), 3, 3))
    out[:, 0, :] = np.transpose(
        np.stack(
            [
                fact * (r_y**2 + r_z**2),
                -fact * r_x * r_y,
                -fact * r_x * r_z,
            ],
        ),
    )

    out[:, 1, :] = np.transpose(
        np.stack([0.0 * fact, fact * r_z, -fact * r_y]),
    )

    out[:, 2, :] = np.transpose(np.stack([r_x, r_y, r_z]))

    if direction == 1:
        out = np.transpose(out, [0, 2, 1])
    elif direction == -1:
        out = np.transpose(out, [0, 1, 2])
    else:
        raise ValueError("Direction must be either 1 or -1.")

    return out


def dsl2gse(
    inp: DataArray, defatt: Union[Dataset, np.ndarray], direction: Optional[int] = 1
):
    r"""Transform time series from MMS's DSL to GSE.

    Parameters
    ----------
    inp : DataArray
        Input time series to convert.
    defatt : Dataset or numpy.ndarray
        Spacecraft attitude.
    direction : {1, -1}, Optional
        Direction of transformation. +1 DSL -> GSE, -1 GSE -> DSL.
        Default is 1.

    Returns
    -------
    out : xarray.DataArray
        Time series of the input field in the new coordinates systems.

    Raises
    ------
    TypeError
        If defatt is not a Dataset or a numpy.ndarray.

    Examples
    --------
    >>> from pyrfu.mms import get_data, load_ancillary, dsl2gse

    Define time interval

    >>> tint = ["2015-05-09T14:00:000", "2015-05-09T17:59:590"]

    Load magnetic field in spacecraft coordinates

    >>> b_xyz = get_data("b_dmpa_fgm_brst_l2", tint, 1)

    Load spacecraft attitude

    >>> defatt = load_ancillary("defatt", tint, 1)

    Transform magnetic field to GSE

    >>> b_gse = dsl2gse(b_xyz, defatt)

    """
    if isinstance(defatt, xr.Dataset):
        x = np.cos(np.deg2rad(defatt.z_dec)) * np.cos(
            np.deg2rad(defatt.z_ra.data),
        )
        y = np.cos(np.deg2rad(defatt.z_dec)) * np.sin(
            np.deg2rad(defatt.z_ra.data),
        )
        z = np.sin(np.deg2rad(defatt.z_dec))
        sax_gei = ts_vec_xyz(defatt.time.data, np.transpose(np.vstack([x, y, z])))
        sax_gse = cotrans(sax_gei, "gei>gse")
        spin_ax_gse = resample(sax_gse, inp, f_s=calc_fs(inp))
        spin_axis = spin_ax_gse.data

    elif isinstance(defatt, np.ndarray) and len(defatt) == 3:
        spin_axis = np.atleast_2d(defatt)

    else:
        raise TypeError("DEFATT/SAX input must be xarray.Dataset or vector")

    # Compute transformation natrix
    transf_mat = _transformation_matrix(spin_axis, direction)

    out_data = np.einsum("kji,ki->kj", transf_mat, inp.data)

    out = inp.copy()
    out.data = out_data
    out.attrs["COORDINATE_SYSTEM"] = "GSE"

    return out
