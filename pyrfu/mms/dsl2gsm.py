#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Optional, Union

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
from .dsl2gse import _transformation_matrix

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def dsl2gsm(
    inp: DataArray, defatt: Union[Dataset, np.ndarray], direction: Optional[int] = 1
) -> DataArray:
    r"""Transform time series from MMS's DSL to GSM.

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
    DataArray
        Time series of the input field in the new coordinates systems.

    Raises
    ------
    TypeError
        If defatt is not xarray.Dataset or numpy.ndarray.

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
        sax_gsm = cotrans(sax_gei, "gei>gsm")
        spin_ax_gsm = resample(sax_gsm, inp, f_s=calc_fs(inp))
        spin_axis = spin_ax_gsm.data

    elif isinstance(defatt, np.ndarray) and len(defatt) == 3:
        spin_axis = np.atleast_2d(defatt)

    else:
        raise TypeError("DEFATT/SAX input must be xarray.Dataset or vector")

    # Compute transformation matrix
    transf_mat = _transformation_matrix(spin_axis, direction)

    out_data = np.einsum("kji,ki->kj", transf_mat, inp.data)

    out = inp.copy()
    out.data = out_data
    out.attrs["COORDINATE_SYSTEM"] = "GSM"

    return out
