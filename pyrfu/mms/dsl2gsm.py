#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports

# 3rd party imports
from typing import Literal

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

# Local imports
from ..pyrf.calc_fs import calc_fs
from ..pyrf.cotrans import cotrans
from ..pyrf.resample import resample
from .dsl2gse import _spin_axis_gei, _transformation_matrix


def dsl2gsm(
    inp: DataArray,
    defatt: Dataset | np.ndarray,
    direction: Literal[1, -1] = 1,
    *,
    axis: Literal["z", "l"] = "z",
) -> DataArray:
    r"""Transform time series from MMS's DSL to GSM.

    Parameters
    ----------
    inp : DataArray
        Input time series to convert.
    defatt : Dataset or numpy.ndarray
        Spacecraft attitude (DEFATT), or the spin axis in GSM as a 3-vector
        (normalised internally).
    direction : {1, -1}, Optional
        Direction of transformation. +1 DSL -> GSM, -1 GSM -> DSL.
        Default is 1.
    axis : {"z", "l"}, Optional
        Axis of DEFATT taken as the spin axis: the body z axis ("z"), as
        irfu-matlab, or the angular momentum ("l"), the z axis of DMPA (see
        `pyrfu.mms.dsl2gse`). Default is "z". Not used when `defatt` is a
        3-vector.

    Returns
    -------
    DataArray
        Time series of the input field in the new coordinates systems.
        ``COORDINATE_SYSTEM`` is set to "GSM" (direction 1) or "DSL"
        (direction -1).

    Raises
    ------
    TypeError
        If defatt is not xarray.Dataset or numpy.ndarray.
    ValueError
        If axis is not "z" or "l".

    Examples
    --------
    >>> from pyrfu.mms import get_data, load_ancillary, dsl2gsm

    Define time interval

    >>> tint = ["2015-05-09T14:00:000", "2015-05-09T17:59:590"]

    Load magnetic field in spacecraft coordinates

    >>> b_xyz = get_data("b_dmpa_fgm_brst_l2", tint, 1)

    Load spacecraft attitude

    >>> defatt = load_ancillary("defatt", tint, 1)

    Transform magnetic field to GSM coordinates

    >>> b_gsm = dsl2gsm(b_xyz, defatt)

    """
    if isinstance(defatt, xr.Dataset):
        sax_gsm = cotrans(_spin_axis_gei(defatt, axis), "gei>gsm")
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
    out.attrs["COORDINATE_SYSTEM"] = "GSM" if direction == 1 else "DSL"

    return out
