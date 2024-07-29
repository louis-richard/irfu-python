#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.pyrf.ts_scalar import ts_scalar

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def calc_ag(p_xyz: DataArray) -> DataArray:
    r"""Compute Che's agyrotropy coefficient.

     Che's agyrotropy is [16]_

    .. math::

        AG^{1/3} = \frac{|\operatorname[det]{\mathbf{P}}
        - \operatorname[det]{\mathbf{P}}|}
        {\operatorname[det]{\mathbf{P}}
        + \operatorname[det]{\mathbf{P}}}


    Parameters
    ----------
    p_xyz : DataArray
        Time series of the pressure tensor

    Returns
    -------
    DataArray
        Time series of the agyrotropy coefficient of the specie.

    Raises
    ------
    TypeError
        If input is not a xarray.DataArray.
    ValueError
        If input is not a time series of a tensor (n_time, 3, 3).

    References
    ----------
    .. [16] H. Che, C. Schiff, G. Le, J. C. Dorelli, B. L. Giles, and T.
            E. Moore (2018), Quantifying the effect of non-Larmor motion
            of electrons on the pres- sure tensor, Phys. Plasmas 25(3),
            032101, doi: https://doi.org/10.1063/1.5016853.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> ic = 1

    Load magnetic field and electron pressure tensor

    >>> b_xyz = mms.get_data("b_gse_fgm_srvy_l2", tint, 1)
    >>> p_xyz_e = mms.get_data("pe_gse_fpi_fast_l2", tint, 1)

    Rotate electron pressure tensor to field aligned coordinates

    >>> p_fac_e_pp = mms.rotate_tensor(p_xyz_e, "fac", b_xyz, "pp")

    Compute agyrotropy coefficient

    >>> ag_e, ag_cr_e = pyrf.calc_ag(p_fac_e_pp)

    """
    # Check input type
    if not isinstance(p_xyz, xr.DataArray):
        raise TypeError("p_xyz must be a xarray.DataArray")

    # Check input shape
    if p_xyz.data.ndim != 3 or p_xyz.shape[1] != 3 or p_xyz.shape[2] != 3:
        raise ValueError("p_xyz must be a time series of a tensor (n_time, 3, 3)")

    # Diagonal and off-diagonal terms
    p_11, p_22, _ = [p_xyz.data[:, 0, 0], p_xyz.data[:, 1, 1], p_xyz.data[:, 2, 2]]
    p_12, p_13, p_23 = [p_xyz.data[:, 0, 1], p_xyz.data[:, 0, 2], p_xyz.data[:, 1, 2]]

    det_p = p_11 * (p_22**2 - p_23**2)
    det_p -= p_12 * (p_12 * p_22 - p_23 * p_13)
    det_p += p_13 * (p_12 * p_23 - p_22 * p_13)

    det_g = p_11 * p_22**2

    agyrotropy = np.abs(det_p - det_g) / (det_p + det_g)
    agyrotropy = ts_scalar(p_xyz.time.data, agyrotropy)

    return agyrotropy
