#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Sequence

# 3rd party imports
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray

# Local imports
from ..pyrf.resample import resample

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.5.1"
__status__ = "Prototype"

# Spacecraft pairs (0-based), in the order of irfu-matlab: 12, 13, 14, 23, 24, 34
_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def mms4_displacement(
    b_mms: Sequence[DataArray], r_mms: Sequence[DataArray]
) -> tuple[DataArray, DataArray, DataArray]:
    r"""Displacements between the MMS spacecraft parallel and perpendicular
    to the magnetic field.

    For each pair of spacecraft :math:`(i, j)`, the separation
    :math:`R_{ij} = R_i - R_j` is decomposed along and across the average
    magnetic field of the pair, :math:`(B_i + B_j) / 2`. Port of
    mms.mms4_displacement in irfu-matlab, without the loading of the data
    from a time interval and without the figure.

    Parameters
    ----------
    b_mms : list of xarray.DataArray
        Time series of the magnetic field of the four spacecraft, in nT.
    r_mms : list of xarray.DataArray
        Time series of the positions of the four spacecraft, in km, in the
        same coordinate system as `b_mms`.

    Returns
    -------
    r_par : xarray.DataArray
        Time series of the separations along B, in km, for the pairs
        12, 13, 14, 23, 24 and 34 (dimension ``pair``).
    r_perp : xarray.DataArray
        Time series of the magnitude of the separations perpendicular to
        B, in km.
    theta_br : xarray.DataArray
        Time series of the angles between B and the separations, in
        degrees.

    Raises
    ------
    ValueError
        If `b_mms` or `r_mms` does not have four time series.

    Notes
    -----
    The fields and positions are resampled to the time line of the field of
    MMS1. For a structure moving at :math:`v_\parallel` along B, the delay
    between spacecraft :math:`i` and :math:`j` is
    :math:`t_i - t_j = R_{\parallel, ij} / v_\parallel`.

    See Also
    --------
    pyrfu.pyrf.c_4_v : Velocity of a discontinuity from four-spacecraft
        timing.

    Examples
    --------
    >>> from pyrfu import mms

    >>> tint = ["2017-07-06T13:54:00.000", "2017-07-06T13:55:00.000"]
    >>> b_mms = [mms.get_data("b_gse_fgm_brst_l2", tint, i) for i in range(1, 5)]
    >>> r_mms = [mms.get_data("r_gse_mec_srvy_l2", tint, i) for i in range(1, 5)]
    >>> r_par, r_perp, theta_br = mms.mms4_displacement(b_mms, r_mms)

    """

    if len(b_mms) != 4 or len(r_mms) != 4:
        raise ValueError("b_mms and r_mms must have the time series of 4 spacecraft")

    b_ref = b_mms[0]
    b_xyz = [resample(b, b_ref).data[:, :3] for b in b_mms]
    r_xyz = [resample(r, b_ref).data[:, :3] for r in r_mms]

    r_par, r_perp, theta_br = [np.zeros((len(b_ref.time), 6)) for _ in range(3)]

    for k, (i, j) in enumerate(_PAIRS):
        r_ij = r_xyz[i] - r_xyz[j]
        b_ij = (b_xyz[i] + b_xyz[j]) / 2
        b_hat = b_ij / np.linalg.norm(b_ij, axis=1, keepdims=True)

        r_par[:, k] = np.sum(r_ij * b_hat, axis=1)
        r_perp[:, k] = np.linalg.norm(r_ij - r_par[:, k, None] * b_hat, axis=1)

        cos_theta = r_par[:, k] / np.linalg.norm(r_ij, axis=1)
        theta_br[:, k] = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    coords = [b_ref.time.data, ["12", "13", "14", "23", "24", "34"]]
    dims = ["time", "pair"]

    r_par = xr.DataArray(r_par, coords=coords, dims=dims, attrs={"UNITS": "km"})
    r_perp = xr.DataArray(r_perp, coords=coords, dims=dims, attrs={"UNITS": "km"})
    theta_br = xr.DataArray(theta_br, coords=coords, dims=dims, attrs={"UNITS": "deg"})

    return r_par, r_perp, theta_br
