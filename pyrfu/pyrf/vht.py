#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import logging

import numpy as np

# Local imports
from .resample import resample
from .ts_vec_xyz import ts_vec_xyz

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logger = logging.getLogger(__name__)


def vht(e, b, no_ez: bool = False):
    r"""Estimate velocity of the De Hoffman-Teller frame from the velocity
    estimate the electric field eht=-vht x b

    The velocity minimises
    :math:`D = \langle |\mathbf{E} + \mathbf{v}_{HT} \times \mathbf{B}|^2
    \rangle`, i.e., solves :math:`K \mathbf{v}_{HT} = \langle \mathbf{E}
    \times \mathbf{B} \rangle` with :math:`K = \langle B^2 I - \mathbf{B}
    \mathbf{B}^T \rangle` [1]_, using the samples where E and B are finite.
    The uncertainty is estimated from :math:`S = D K^{-1} / (2M - 3)` for M
    samples [1]_, which assumes that the noise in E is perpendicular to B.

    Unlike irf_vht.m, K and :math:`\langle \mathbf{E} \times \mathbf{B}
    \rangle` are averaged over the same samples, so that data gaps do not
    bias the velocity.

    Parameters
    ----------
    e : xarray.DataArray
        Time series of the electric field [mV/m].
    b : xarray.DataArray
        Time series of the magnetic field [nT].
    no_ez : boolean, Optional
        If True assumed no Ez. Default is False.

    Returns
    -------
    vht : numpy.ndarray
        De Hoffman Teller frame velocity [km/s].
    e_ht : xarray.DataArray
        Time series of the electric field in the De Hoffman frame,
        -v_ht x B [mV/m].
    dv_ht : numpy.ndarray
        Error of De Hoffman Teller frame [km/s].

    References
    ----------
    .. [1]  Khrabrov, A. V., and B. U. O. Sonnerup (1998), DeHoffmann-Teller
            analysis, in Analysis Methods for Multi-Spacecraft Data, edited by
            G. Paschmann and P. W. Daly, pp. 221-248, Int. Space Sci. Inst.,
            Bern.

    """

    # Resample magnetic field to electric field sampling (usually higher)
    if not np.array_equal(e.time.data, b.time.data):
        b = resample(b, e)

    # assume only Ex and Ey: put z component to 0 when using only Ex and Ey
    n_comp = 2 if no_ez else 3

    # Samples where E and B are finite, used for all the averages
    valid = np.all(np.isfinite(b.data), axis=1)
    valid &= np.all(np.isfinite(e.data[:, :n_comp]), axis=1)
    n_valid = int(np.sum(valid))

    b_v = b.data[valid].astype(np.float64)
    e_v = e.data[valid].astype(np.float64)
    e_v[:, n_comp:] = 0.0

    # <Bi Bj>
    b_ij = np.einsum("ti,tj->ij", b_v, b_v) / n_valid

    if no_ez:
        k_mat = np.array(
            [
                [b_ij[2, 2], 0, -b_ij[0, 2]],
                [0, b_ij[2, 2], -b_ij[1, 2]],
                [-b_ij[0, 2], -b_ij[1, 2], b_ij[0, 0] + b_ij[1, 1]],
            ],
        )
    else:
        k_mat = np.trace(b_ij) * np.eye(3) - b_ij

    exb_avg = np.mean(np.cross(e_v, b_v), axis=0)

    v_ht = np.linalg.solve(k_mat, exb_avg) * 1e3  # 9.12 in ISSI book

    v_ht_hat = v_ht / np.linalg.norm(v_ht, keepdims=True)

    logger.info(
        "v_ht =%(v_mag)7.4f * %(v_vec)s km/s",
        {"v_mag": np.linalg.norm(v_ht), "v_vec": np.array_str(v_ht_hat)},
    )

    # Calculate the goodness of the Hoffman Teller frame
    # e_ht = e_vxb(v_ht, b)
    e_ht = ts_vec_xyz(b.time.data, -1e-3 * np.cross(v_ht, b.data))

    e_ht_p = e_ht.data[valid]
    e_ht_p[:, n_comp:] = 0.0

    delta_e = e_v - e_ht_p

    poly_fit = np.polyfit(e_ht_p.ravel(), e_v.ravel(), 1)
    corr_coeff = np.corrcoef(e_ht_p.ravel(), e_v.ravel())

    logger.info(
        "slope = %(slope)6.4f, offs = %(offset)6.4f, cc = %(cc)6.4f",
        {"slope": poly_fit[0], "offset": poly_fit[1], "cc": corr_coeff[0, 1]},
    )

    d_ht = np.sum(delta_e**2) / n_valid
    s_mat = d_ht / (2 * n_valid - 3) * np.linalg.inv(k_mat)
    dv_ht = np.sqrt(np.diag(s_mat)) * 1e3

    dv_ht_hat = dv_ht / np.linalg.norm(dv_ht)

    logger.info(
        "dv_ht =%(dv_mag)7.4f * %(dv_vec)s km/s",
        {"dv_mag": np.linalg.norm(dv_ht), "dv_vec": np.array_str(dv_ht_hat)},
    )

    return v_ht, e_ht, dv_ht
