#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

import xarray as xr
import numpy as np


def mva(inp, flag="mvar"):
    """Compute the minimum variance frame.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the quantity to find minimum variance frame.

    flag : str {"mvar", "<bn>=0", "td"}
        Constrain.

    Returns
    -------
    out : xarray.DataArray
        Time series of the input quantity in LMN coordinates.

    l : ndarray
        Eigenvalues l[0]>l[1]>l[2].

    lmn : ndarray
        Eigenvectors LMN coordinates.

    See also
    --------
    pyrfu.pyrf.new_xyz

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Spacecraft index

    >>> mms_id = 1

    Load magnetic field

    >>> b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)

    Compute MVA frame

    >>> b_lmn, lamb, frame = pyrf.mva(b_xyz)

    """

    inp_data = inp.data

    inp_m = np.mean(inp_data, 0)

    if flag in ["mvar", "<bn>=0"]:
        m_mu_nu_m = np.mean(inp_data[:, [0, 1, 2, 0, 0, 1]] * inp_data[:, [0, 1, 2, 1, 2, 2]], 0)
        m_mu_nu_m -= inp_m[[0, 1, 2, 0, 0, 1]] * inp_m[[0, 1, 2, 1, 2, 2]]

    elif flag.lower() == "td":
        m_mu_nu_m = np.mean(inp_data[:, [0, 1, 2, 0, 0, 1]] * inp_data[:, [0, 1, 2, 1, 2, 2]], 0)

    else:
        raise ValueError("invalid flag")

    m_mu_nu = np.array([m_mu_nu_m[[0, 3, 4]], m_mu_nu_m[[3, 1, 5]], m_mu_nu_m[[4, 5, 2]]])

    # Compute eigenvalues and eigenvectors
    [l, lmn] = np.linalg.eig(m_mu_nu)

    # Sort eigenvalues
    idx = l.argsort()[::-1]
    l, lmn = [l[idx], lmn[:, idx]]

    # ensure that the frame is right handed
    lmn[:, 2] = np.cross(lmn[:, 0], lmn[:, 1])

    if flag.lower() == "<bn>=0":
        inp_mvar_mean = np.mean(
            np.sum(np.tile(inp_data, (3, 1, 1)) * np.transpose(
                np.tile(lmn, (inp_data.shape[0], 1, 1)), (2, 0, 1)), 1), 1)

        a = np.sum(inp_mvar_mean ** 2)

        b = -(l[1] + l[2]) * inp_mvar_mean[0] ** 2
        b -= (l[0] + l[2]) * inp_mvar_mean[1] ** 2
        b -= (l[0] + l[1]) * inp_mvar_mean[2] ** 2

        c = l[1] * l[2] * inp_mvar_mean[0] ** 2
        c += l[0] * l[2] * inp_mvar_mean[1] ** 2
        c += l[0] * l[1] * inp_mvar_mean[2] ** 2

        r = np.roots([a, b, c])

        l_min = np.min(r)

        n = inp_mvar_mean / (l - l_min)
        n /= np.linalg.norm(n, keepdims=True)
        n = np.matmul(lmn, n)

        bn = np.sum(inp_data * np.tile(n, (inp_data.shape[0], 1)), axis=1)

        inp_data_2 = inp_data - np.tile(bn, (3, 1)).T * np.tile(n, (inp_data.shape[0], 1))

        inp_data_2_m = np.mean(inp_data_2, 0)

        m_mu_nu_m = np.mean(inp_data_2[:, [0, 1, 2, 0, 0, 1]] * inp_data_2[:, [0, 1, 2, 1, 2, 2]],
                            0)
        m_mu_nu_m -= inp_data_2_m[[0, 1, 2, 0, 0, 1]] * inp_data_2_m[[0, 1, 2, 1, 2, 2]]

        m_mu_nu = np.array([m_mu_nu_m[[0, 3, 4]], m_mu_nu_m[[3, 1, 5]], m_mu_nu_m[[4, 5, 2]]])

        l, lmn = np.linalg.eig(m_mu_nu)

        idx = l.argsort()[::-1]

        l, lmn = [l[idx], lmn[:, idx]]

        l[2], lmn[:, 2] = [l_min, np.cross(lmn[:, 0], lmn[:, 1])]

    elif flag.lower() == "td":
        ln = l[2]
        bn = np.sum(inp_data * np.tile(lmn[:, 2], (inp_data.shape[0], 1)), axis=1)

        inp_data_2 = inp_data - np.tile(bn, (3, 1)).T * np.tile(lmn[:, 2], (inp_data.shape[0], 1))

        inp_data_2_m = np.mean(inp_data_2, 0)

        m_mu_nu_m = np.mean(inp_data_2[:, [0, 1, 2, 0, 0, 1]] * inp_data_2[:, [0, 1, 2, 1, 2, 2]],
                            0)
        m_mu_nu_m -= inp_data_2_m[[0, 1, 2, 0, 0, 1]] * inp_data_2_m[[0, 1, 2, 1, 2, 2]]

        m_mu_nu = np.array([m_mu_nu_m[[0, 3, 4]], m_mu_nu_m[[3, 1, 5]], m_mu_nu_m[[4, 5, 2]]])

        l, lmn = np.linalg.eig(m_mu_nu)

        idx = l.argsort()[::-1]

        l, lmn = [l[idx], lmn[:, idx]]

        l[2], lmn[:, 2] = [ln, np.cross(lmn[:, 0], lmn[:, 1])]

    out_data = (lmn.T @ inp_data.T).T

    out = xr.DataArray(out_data, coords=inp.coords, dims=inp.dims)

    return out, l, lmn
