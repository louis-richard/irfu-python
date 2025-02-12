#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import logging

import numpy as np

# Local imports
from .resample import resample
from .ts_vec_xyz import ts_vec_xyz

# from .e_vxb import e_vxb


__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def vht(e, b, no_ez: bool = False):
    r"""Estimate velocity of the De Hoffman-Teller frame from the velocity
    estimate the electric field eht=-vht x b

    Parameters
    ----------
    e : xarray.DataArray
        Time series of the electric field.
    b : xarray.DataArray
        Time series of the magnetic field.
    no_ez : boolean, Optional
        If True assumed no Ez. Default is False.

    Returns
    -------
    vht : numpy.ndarray
        De Hoffman Teller frame velocity [km/s].
    vht : xarray.DataArray
        Time series of the electric field in the De Hoffman frame.
    dv_ht : numpy.ndarray
        Error of De Hoffman Teller frame.

    """

    n_samples = len(e)

    # Resample magnetic field to electric field sampling (usually higher)

    if n_samples != len(b):
        b = resample(b, e)

    p = np.zeros(6)

    p[0] = np.sum(b[:, 0].data * b[:, 0].data) / n_samples  # Bx*Bx
    p[1] = np.sum(b[:, 0].data * b[:, 1].data) / n_samples  # Bx*By
    p[2] = np.sum(b[:, 0].data * b[:, 2].data) / n_samples  # Bx*Bz
    p[3] = np.sum(b[:, 1].data * b[:, 1].data) / n_samples  # By*By
    p[4] = np.sum(b[:, 1].data * b[:, 2].data) / n_samples  # By*Bz
    p[5] = np.sum(b[:, 2].data * b[:, 2].data) / n_samples  # Bz*Bz

    # assume only Ex and Ey
    if no_ez:
        e[:, 2] *= 0  # put z component to 0 when using only Ex and Ey

        k_mat = np.array(
            [[p[5], 0, -p[2]], [0, p[5], -p[4]], [-p[2], -p[4], p[0] + p[3]]],
        )
    else:
        k_mat = np.array(
            [
                [p[3] + p[5], -p[1], -p[2]],
                [-p[1], p[0] + p[5], -p[4]],
                [-p[2], -p[4], p[0] + p[3]],
            ],
        )

    exb = np.cross(e, b)

    ind_data = np.where(~np.isnan(exb[:, 0].data))[0]

    # revised by Wenya LI; 2015-11-21, wyli @ irfu
    exb_avg = np.sum(exb[ind_data, :], axis=0) / n_samples

    # averExB=sum(ExB(indData).data,1)/nSamples;
    # end revise.
    v_ht = np.linalg.solve(k_mat, exb_avg.T) * 1e3  # 9.12 in ISSI book

    v_ht_hat = v_ht / np.linalg.norm(v_ht, keepdims=True)

    logging.info(
        "v_ht =%(v_mag)7.4f * %(v_vec)s km/s",
        {"v_mag": np.linalg.norm(v_ht), "v_vec": np.array_str(v_ht_hat)},
    )

    # Calculate the goodness of the Hoffman Teller frame
    # e_ht = e_vxb(v_ht, b)
    e_ht = ts_vec_xyz(b.time.data, -1e-3 * np.cross(v_ht, b.data))

    if no_ez:
        e_p, e_ht_p = [e[ind_data], e_ht[ind_data]]
        e_p.data[:, 2], e_ht_p.data[:, 2] = [0, 0]
    else:
        e_p, e_ht_p = [e[ind_data], e_ht[ind_data]]

    delta_e = e_p.data - e_ht_p.data

    poly_fit = np.polyfit(
        e_ht_p.data.reshape([len(e_ht_p) * 3]),
        e_p.data.reshape([len(e_p) * 3]),
        1,
    )
    corr_coeff = np.corrcoef(
        e_ht_p.data.reshape([len(e_ht_p) * 3]),
        e_p.data.reshape([len(e_p) * 3]),
    )

    logging.info(
        "slope = %(slope)6.4f, offs = %(offset)6.4f, cc = %(cc)6.4f",
        {"slope": poly_fit[0], "offset": poly_fit[1], "cc": corr_coeff[0, 1]},
    )

    dv_ht = np.sum(np.sum(delta_e**2)) / len(ind_data)
    s_mat = (dv_ht / (2 * len(ind_data) - 3)) / k_mat
    dv_ht = np.sqrt(np.diag(s_mat)) * 1e3

    dv_ht_hat = dv_ht / np.linalg.norm(dv_ht)

    logging.info(
        "dv_ht =%(dv_mag)7.4f * %(dv_vec)s km/s",
        {"dv_mag": np.linalg.norm(dv_ht), "dv_vec": np.array_str(dv_ht_hat)},
    )

    return v_ht, e_ht, dv_ht
