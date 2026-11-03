#!/usr/bin/env python
# -*- coding: utf-8 -*-


from typing import Tuple

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

# Local imports
from pyrfu.pyrf._units import _attrs, _units
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.ts_scalar import ts_scalar
from pyrfu.pyrf.ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def dec_par_perp(
    inp: DataArray, b_bgd: DataArray, flag_spin_plane: bool = False
) -> Tuple[DataArray, DataArray, DataArray]:
    r"""Decomposes a vector into par/perp to B components. If flagspinplane
    decomposes components to the projection of ``b0`` into the XY plane.
    ``alpha`` gives the angle between ``b0`` and the XY. plane.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the field to decompose (3 components; with
        flag_spin_plane, 2 or 3, of which x and y are used).
    b_bgd : xarray.DataArray
        Time series of the background magnetic field (3 components).
    flag_spin_plane : bool, Optional
        Flag if True gives the projection in XY plane.

    Returns
    -------
    a_para : xarray.DataArray
        Time series of the input field parallel to the background magnetic
        field.
    a_perp : xarray.DataArray
        Time series of the input field perpendicular to the background
        magnetic field.
    alpha : xarray.DataArray
        Time series of the angle between the background magnetic field and
        the XY plane, in radians (irfu-matlab ``irf_dec_parperp`` returns it
        in degrees), with flag_spin_plane; zero otherwise.

    Raises
    ------
    TypeError
        If inp or b_bgd is not a DataArray, or flag_spin_plane not a bool.
    ValueError
        If inp or b_bgd does not have the components above.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field (FGM) and electric field (EDP)

    >>> b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, mms_id)
    >>> e_xyz = mms.get_data("E_gse_edp_brst_l2", tint, mms_id)

    Decompose e_xyz into parallel and perpendicular to b_xyz components

    >>> e_para, e_perp, _ = pyrf.dec_par_perp(e_xyz, b_xyz)

    """

    # Check arguments types
    if not isinstance(inp, xr.DataArray) or not isinstance(b_bgd, xr.DataArray):
        raise TypeError("inp and b_bgd must be xarray.DataArray")

    if not isinstance(flag_spin_plane, bool):
        raise TypeError("flag_spin_plane must be a bool")

    # Check inp and b_bgd shapes (the components were not checked: an inp of
    # 2 components failed in a broadcast); in the spin plane, inp may have
    # only x and y (as irfu-matlab irf_dec_parperp)
    n_comps = (2, 3) if flag_spin_plane else (3,)

    if inp.ndim != 2 or inp.shape[1] not in n_comps:
        raise ValueError(
            f"inp must be a vector of {' or '.join(map(str, n_comps))} components"
        )

    if b_bgd.ndim != 2 or b_bgd.shape[1] != 3:
        raise ValueError("b_bgd must be a vector of 3 components")

    # Resample background magnetic field to input
    b_bgd = resample(b_bgd, inp)

    # Get times and data
    inp_time = inp.time.data
    inp_data = inp.data
    b_bgd_data = b_bgd.data

    if not flag_spin_plane:
        b_mag = np.linalg.norm(b_bgd_data, axis=1, keepdims=True)

        indices = np.where(b_mag < 1e-3)[0]

        if indices.size > 0:
            b_mag[indices] = 1e-3 * np.ones((len(indices), 1))

        b_hat = b_bgd_data / b_mag

        a_para_arr = np.sum(b_hat * inp_data, axis=-1)
        a_para = ts_scalar(inp_time, a_para_arr)

        a_perp_arr = inp_data - (
            b_hat * np.tile(a_para_arr[:, np.newaxis], (1, inp_data.shape[1]))
        )
        a_perp = ts_vec_xyz(inp_time, a_perp_arr)

        alpha_arr = np.zeros_like(a_para_arr)
        alpha = ts_scalar(inp_time, alpha_arr)
    else:
        b_tot = np.sqrt(b_bgd_data[:, 0] ** 2 + b_bgd_data[:, 1] ** 2)
        b_hat = b_bgd_data / b_tot[:, np.newaxis]

        a_para_arr = inp_data[:, 0] * b_hat[:, 0] + inp_data[:, 1] * b_hat[:, 1]
        a_para = ts_scalar(inp_time, a_para_arr)

        a_perp_arr = inp_data[:, 0] * b_hat[:, 1] - inp_data[:, 1] * b_hat[:, 0]
        a_perp = ts_scalar(inp_time, a_perp_arr)

        alpha_arr = np.arctan2(b_bgd_data[:, 2], b_tot)
        alpha = ts_scalar(inp_time, alpha_arr)

    # Components in the units of inp, alpha in rad (they had none)
    for out in [a_para, a_perp]:
        out.attrs = _attrs(out.attrs, _units(inp))

    alpha.attrs["UNITS"] = "rad"

    return a_para, a_perp, alpha
