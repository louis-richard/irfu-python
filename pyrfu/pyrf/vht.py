#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vht.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr

from .resample import resample
from .e_vxb import e_vxb


def vht(e=None, b=None, flag=1):
    """
    Estimate velocity of the De Hoffman-Teller frame from the velocity estimate the electric field
    eht=-vht x b

    Parameters
    ----------
    e : xarray.DataArray
        Time series of the electric field.

    b : xarray.DataArray
        Time series of the magnetic field.

    flag : int
        If 2 assumed no Ez.

    Returns
    -------
    vht : numpy.ndarray
        De Hoffman Teller frame velocity [km/s].

    vht : xarray.DataArray
        Time series of the electric field in the De Hoffman frame.

    dv_ht : numpy.ndarray
        Error of De Hoffman Teller frame.

    """

    assert e is not None and isinstance(e, xr.DataArray)
    assert b is not None and isinstance(b, xr.DataArray)

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
    if flag == 2:
        e[:, 2] *= 0     # put z component to 0 when using only Ex and Ey

        k_mat = np.array([[p[5], 0, -p[2]], [0, p[5], -p[4]], [-p[2], -p[4], p[0] + p[3]]])
        u_msg = "De Hoffman-Teller frame is calculated using 2 components of E=(Ex,Ey,0)"
    else:
        k_mat = np.array(
            [[p[3] + p[5], -p[1], -p[2]], [-p[1], p[0] + p[5], -p[4]], [-p[2], -p[4], p[0] + p[3]]])
        u_msg = "De Hoffman-Teller frame is calculated using all 3 components of E=(Ex,Ey,Ez)"

    exb = np.cross(e, b)

    ind_data = np.where(~np.isnan(exb[:, 0].data))[0]

    # revised by Wenya LI; 2015-11-21, wyli @ irfu
    tmp1 = exb[ind_data]
    exb_avg = np.sum(tmp1, axis=0) / n_samples
    # averExB=sum(ExB(indData).data,1)/nSamples;
    # end revise.
    v_ht = np.linalg.solve(k_mat, exb_avg.T) * 1e3  # 9.12 in ISSI book

    v_ht_hat = v_ht/np.linalg.norm(v_ht, keepdims=True)

    print(u_msg)
    print("v_ht ={:7.4f} * {} = {} km/s".format(np.linalg.norm(v_ht), np.array_str(v_ht_hat),
                                                np.array_str(v_ht)))

    # Calculate the goodness of the Hoffman Teller frame
    e_ht = e_vxb(v_ht, b)

    if flag == 2:
        e_p, e_ht_p = [e[ind_data], e_ht[ind_data]]
        e_p.data[:, 2], e_ht_p.data[:, 2] = [0, 0]
    else:
        e_p, e_ht_p = [e[ind_data], e_ht[ind_data]]

    delta_e = e_p.data - e_ht_p.data

    poly_fit = np.polyfit(e_ht_p.data.reshape([len(e_ht_p) * 3]), e_p.data.reshape([len(e_p) * 3]),
                          1)
    corr_coeff = np.corrcoef(e_ht_p.data.reshape([len(e_ht_p) * 3]),
                             e_p.data.reshape([len(e_p) * 3]))

    print("slope = {p[0]:6.4f}, offs = {p[1]:6.4f}".format(p=poly_fit))
    print("cc = {:6.4f}".format(corr_coeff[0, 1]))

    dv_ht = np.sum(np.sum(delta_e ** 2)) / len(ind_data)
    s_mat = (dv_ht / (2 * len(ind_data) - 3)) / k_mat
    dv_ht = np.sqrt(np.diag(s_mat)) * 1e3

    dv_ht_hat = dv_ht / np.linalg.norm(dv_ht)

    print("dv_ht ={:7.4f} * {} = {} km/s".format(np.linalg.norm(dv_ht), np.array_str(dv_ht_hat),
                                                 np.array_str(dv_ht)))

    return v_ht, e_ht, dv_ht
