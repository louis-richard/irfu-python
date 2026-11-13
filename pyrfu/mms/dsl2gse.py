#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
from typing import Literal

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

# Local imports
from ..pyrf.calc_fs import calc_fs
from ..pyrf.cotrans import cotrans
from ..pyrf.resample import resample
from ..pyrf.ts_vec_xyz import ts_vec_xyz


def _transformation_matrix(spin_axis: np.ndarray, direction: int) -> np.ndarray:
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
        ``COORDINATE_SYSTEM`` is set to "GSE" (direction 1) or "DSL"
        (direction -1).

    Raises
    ------
    ValueError
        If direction is not -1 or 1.

    """
    # The rotation assumes a unit spin axis
    spin_axis = spin_axis / np.linalg.norm(spin_axis, axis=1, keepdims=True)
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


def _spin_axis_gei(defatt: Dataset, axis: str) -> DataArray:
    r"""Time series of the spin axis in GEI from the right ascension and
    declination of DEFATT: the body z axis (z_ra, z_dec) or the angular
    momentum (l_ra, l_dec). Shared by dsl2gse and dsl2gsm."""
    if axis not in ("z", "l"):
        raise ValueError(f'axis must be "z" or "l", not {axis!r}')

    r_a = np.deg2rad(defatt[f"{axis}_ra"].data)
    dec = np.deg2rad(defatt[f"{axis}_dec"].data)
    vec = np.transpose(
        np.vstack([np.cos(dec) * np.cos(r_a), np.cos(dec) * np.sin(r_a), np.sin(dec)])
    )

    return ts_vec_xyz(defatt.time.data, vec)


def dsl2gse(
    inp: DataArray,
    defatt: Dataset | np.ndarray,
    direction: Literal[1, -1] = 1,
    *,
    axis: Literal["z", "l"] = "z",
):
    r"""Transform time series from MMS's DSL to GSE.

    Parameters
    ----------
    inp : DataArray
        Input time series to convert.
    defatt : Dataset or numpy.ndarray
        Spacecraft attitude (DEFATT), or the spin axis in GSE as a 3-vector
        (normalised internally).
    direction : {1, -1}, Optional
        Direction of transformation. By convention: +1 DSL -> GSE, -1 GSE -> DSL.
        Default is 1.
    axis : {"z", "l"}, Optional
        Axis of DEFATT taken as the spin axis: the body z axis ("z", z_ra and
        z_dec), as irfu-matlab mms_dsl2gse, or the angular momentum ("l",
        l_ra and l_dec), the z axis of DMPA (MMS CMAD): for the FGM DMPA
        data of MMS1 on 2017-07-06, the largest difference from the GSE data
        of the files is 1.3e-3 of B with "z" and 2.0e-4 with "l". Default is
        "z". Not used when `defatt` is a 3-vector.

    Returns
    -------
    out : xarray.DataArray
        Time series of the input field in the new coordinates systems.

    Raises
    ------
    TypeError
        If defatt is not a Dataset or a numpy.ndarray.
    ValueError
        If axis is not "z" or "l".

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
        sax_gse = cotrans(_spin_axis_gei(defatt, axis), "gei>gse")
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
    out.attrs["COORDINATE_SYSTEM"] = "GSE" if direction == 1 else "DSL"

    return out
