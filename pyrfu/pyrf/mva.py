#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def mva(inp, flag: str = "mvar"):
    r"""Compute the minimum variance frame.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the quantity to find minimum variance frame.
    flag : {"mvar", "<bn>=0", "td"}, Optional
        Constrain. Default is "mvar".

    Returns
    -------
    out : xarray.DataArray
        Time series of the input quantity in LMN coordinates.
    l : numpy.ndarray
        Eigenvalues l[0] > l[1] > l[2].
    lmn : numpy.ndarray
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

    assert flag.lower() in ["mvar", "<bn>=0", "td"], "invalid method!!"

    inp_data = inp.data
    n_t = inp_data.shape[0]

    idx_1, idx_2 = [[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]

    if flag in ["mvar", "<bn>=0"]:
        m_mu_nu_m = np.mean(inp_data[:, idx_1] * inp_data[:, idx_2], 0)
        m_mu_nu_m -= np.mean(inp_data, 0)[idx_1] * np.mean(inp_data, 0)[idx_2]

    else:
        m_mu_nu_m = np.mean(inp_data[:, idx_1] * inp_data[:, idx_2], 0)

    m_mu_nu = np.array(
        [m_mu_nu_m[[0, 3, 4]], m_mu_nu_m[[3, 1, 5]], m_mu_nu_m[[4, 5, 2]]],
    )

    # Compute eigenvalues and eigenvectors
    [lamb, lmn] = np.linalg.eig(m_mu_nu)

    # Sort eigenvalues
    lamb, lmn = [lamb[lamb.argsort()[::-1]], lmn[:, lamb.argsort()[::-1]]]

    # ensure that the frame is right handed
    lmn[:, 2] = np.cross(lmn[:, 0], lmn[:, 1])

    if flag.lower() == "<bn>=0":
        inp_mvar_mean = np.mean(
            np.sum(
                np.tile(inp_data, (3, 1, 1))
                * np.transpose(np.tile(lmn, (n_t, 1, 1)), (2, 0, 1)),
                1,
            ),
            1,
        )

        coeffs = np.zeros(3)
        coeffs[0] = np.sum(inp_mvar_mean**2)

        coeffs[1] = -(lamb[1] + lamb[2]) * inp_mvar_mean[0] ** 2
        coeffs[1] -= (lamb[0] + lamb[2]) * inp_mvar_mean[1] ** 2
        coeffs[1] -= (lamb[0] + lamb[1]) * inp_mvar_mean[2] ** 2

        coeffs[2] = lamb[1] * lamb[2] * inp_mvar_mean[0] ** 2
        coeffs[2] += lamb[0] * lamb[2] * inp_mvar_mean[1] ** 2
        coeffs[2] += lamb[0] * lamb[1] * inp_mvar_mean[2] ** 2

        roots = np.roots(coeffs)

        l_min = np.min(roots)

        n_vec = inp_mvar_mean / (lamb - l_min)
        n_vec /= np.linalg.norm(n_vec, keepdims=True)
        n_vec = np.matmul(lmn, n_vec)

        b_vec = np.sum(inp_data * np.tile(n_vec, (n_t, 1)), axis=1)

        inp_data_2 = inp_data.copy()
        inp_data_2 -= np.tile(b_vec, (3, 1)).T * np.tile(n_vec, (n_t, 1))

        inp_data_2_m = np.mean(inp_data_2, 0)

        m_mu_nu_m = np.mean(inp_data_2[:, idx_1] * inp_data_2[:, idx_2], 0)
        m_mu_nu_m -= inp_data_2_m[idx_1] * inp_data_2_m[idx_2]

        m_mu_nu = np.array(
            [m_mu_nu_m[[0, 3, 4]], m_mu_nu_m[[3, 1, 5]], m_mu_nu_m[[4, 5, 2]]],
        )

        lamb, lmn = np.linalg.eig(m_mu_nu)

        lamb, lmn = [lamb[lamb.argsort()[::-1]], lmn[:, lamb.argsort()[::-1]]]

        # Force the maximum variance direction to be positive
        # lmn[:, 0] *= np.sign(lmn[np.argmax(lmn[:, 0]), 0])
        # lamb[2], lmn[:, 2] = [l_min, np.cross(lmn[:, 0], lmn[:, 1])]

    elif flag.lower() == "td":
        l_min = lamb[2]
        b_vec = np.sum(inp_data * np.tile(lmn[:, 2], (n_t, 1)), axis=1)

        inp_data_2 = inp_data.copy()
        inp_data_2 -= np.tile(b_vec, (3, 1)).T * np.tile(lmn[:, 2], (n_t, 1))

        inp_data_2_m = np.mean(inp_data_2, 0)

        m_mu_nu_m = np.mean(inp_data_2[:, idx_1] * inp_data_2[:, idx_2], 0)
        m_mu_nu_m -= inp_data_2_m[idx_1] * inp_data_2_m[idx_2]

        m_mu_nu = np.array(
            [m_mu_nu_m[[0, 3, 4]], m_mu_nu_m[[3, 1, 5]], m_mu_nu_m[[4, 5, 2]]],
        )

        lamb, lmn = np.linalg.eig(m_mu_nu)

        lamb, lmn = [lamb[lamb.argsort()[::-1]], lmn[:, lamb.argsort()[::-1]]]

        lamb[2], lmn[:, 2] = [l_min, np.cross(lmn[:, 0], lmn[:, 1])]
    else:
        pass

    out_data = (lmn.T @ inp_data.T).T

    out = xr.DataArray(out_data, coords=inp.coords, dims=inp.dims)

    return out, lamb, lmn
