#!/usr/bin/env python

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


def calc_agyro(p_xyz: DataArray) -> DataArray:
    r"""Compute agyrotropy coefficient as

    .. math::

        A\Phi = \frac{|P_{\perp 1} - P_{\perp 2}|}{P_{\perp 1}
        + P_{\perp 2}}


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

    >>> p_fac_e_qq = mms.rotate_tensor(p_xyz_e, "fac", b_xyz, "qq")

    Compute agyrotropy coefficient

    >>> agyro_e = pyrf.calc_agyro(p_fac_e_qq)

    """
    # Check input type
    if not isinstance(p_xyz, xr.DataArray):
        raise TypeError("p_xyz must be a xarray.DataArray")

    # Check input shape
    if p_xyz.data.ndim != 3 or p_xyz.shape[1] != 3 or p_xyz.shape[2] != 3:
        raise ValueError("p_xyz must be a time series of a tensor")

    # Parallel and perpendicular components
    p_perp_1, p_perp_2 = [p_xyz.data[:, 1, 1], p_xyz.data[:, 2, 2]]

    agyrotropy = np.abs(p_perp_1 - p_perp_2) / (p_perp_1 + p_perp_2)
    agyrotropy = ts_scalar(p_xyz.time.data, agyrotropy)

    return agyrotropy
