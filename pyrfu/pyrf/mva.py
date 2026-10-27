#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from xarray.core.dataarray import DataArray

# Local imports
from .new_xyz import new_xyz

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def _eig(m_mu_nu):
    r"""Eigenvalues and eigenvectors of the (real, symmetric) variance matrix.

    NumPy >= 2.5 returns complex arrays from np.linalg.eig; the imaginary parts
    are zero here, so the real parts are the values given by older versions.

    """
    lamb, lmn = np.linalg.eig(m_mu_nu)

    return np.real(lamb), np.real(lmn)


def _variance_frame(data, centred: bool = True):
    r"""Eigenvalues (decreasing) and eigenvectors (columns, right-handed) of
    the variance matrix <B_i B_j> - <B_i> <B_j> of data (time, 3), or
    <B_i B_j> if not centred, the averages ignoring NaN, as irf_minvar."""
    idx_1, idx_2 = [[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    m_mu_nu_m = np.nanmean(data[:, idx_1] * data[:, idx_2], 0)

    if centred:
        m_mu_nu_m -= np.nanmean(data, 0)[idx_1] * np.nanmean(data, 0)[idx_2]

    m_mu_nu = np.array(
        [m_mu_nu_m[[0, 3, 4]], m_mu_nu_m[[3, 1, 5]], m_mu_nu_m[[4, 5, 2]]],
    )

    # Compute eigenvalues and eigenvectors
    lamb, lmn = _eig(m_mu_nu)

    # Sort eigenvalues
    lamb, lmn = [lamb[lamb.argsort()[::-1]], lmn[:, lamb.argsort()[::-1]]]

    # ensure that the frame is right handed
    lmn[:, 2] = np.cross(lmn[:, 0], lmn[:, 1])

    return lamb, lmn


def mva(inp, flag: str = "mvar"):
    r"""Compute the minimum variance frame, as irfu-matlab irf_minvar.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the quantity to find minimum variance frame (time, 3).
    flag : {"mvar", "<bn>=0", "td"}, Optional
        Constrain (case-insensitive). Default is "mvar".

    Returns
    -------
    out : xarray.DataArray
        Time series of the input quantity in LMN coordinates (attributes of the
        input kept, with ``COORDINATE_SYSTEM`` set to "lmn").
    l : numpy.ndarray
        Variances along L, M and N, l[0] > l[1] > l[2] with "mvar". With
        "<bn>=0", l[2] is the minimum variance under the constraint, and with
        "td" the minimum variance without it, as irf_minvar.
    lmn : numpy.ndarray
        L, M and N unit vectors as the columns, lmn[:, 0] being L (rows in
        irf_minvar), a right-handed frame.

    Raises
    ------
    TypeError
        If inp is not a xarray.DataArray.
    ValueError
        If inp does not have 3 components or if flag is not one of the above.

    See also
    --------
    pyrfu.pyrf.new_xyz

    Notes
    -----
    * "mvar" uses the variance matrix <B_i B_j> - <B_i> <B_j>.
    * "<bn>=0" minimises the variance of B.n with the constraint <B.n> = 0:
      the variance l_min = n.M.n is the smallest root of a quadratic
      polynomial in the variances and the averages of the "mvar" frame [1]_.
      L and M are those of the field with B.n removed.
    * "td" (tangential discontinuity) minimises <(B.n)^2> instead, with the
      matrix <B_i B_j>; L and M are those of the field with B.n removed.
    * The averages ignore the NaN samples (each element of the variance
      matrix uses the samples where its two components are finite), as
      irf_minvar.

    References
    ----------
    .. [1] Sonnerup, B. U. Ö., and Scheible, M. (1998), Minimum and maximum
           variance analysis, in Analysis Methods for Multi-Spacecraft Data,
           ISSI Scientific Report SR-001, 185-220.

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

    if not isinstance(inp, DataArray):
        raise TypeError("inp must be a xarray.DataArray")

    if inp.ndim != 2 or inp.shape[1] != 3:
        raise ValueError("inp must be a time series of 3D vectors (time, 3)")

    if not isinstance(flag, str) or flag.lower() not in ["mvar", "<bn>=0", "td"]:
        raise ValueError(f'flag must be "mvar", "<bn>=0" or "td", not {flag!r}')

    flag = flag.lower()

    inp_data = inp.data
    n_t = inp_data.shape[0]

    lamb, lmn = _variance_frame(inp_data, flag != "td")

    if flag == "<bn>=0":
        # Averages of the field along L, M and N (they were summed over time
        # and averaged over the components, n_t / 3 times the averages, which
        # gives the same roots and normal, but NaN with a NaN sample)
        inp_mvar_mean = np.nanmean(inp_data @ lmn, axis=0)

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

        # L and M of the field without its normal component, and the
        # constrained minimum variance (it was the variance along n of that
        # field, zero)
        lamb, lmn = _variance_frame(inp_data_2)
        lamb[2] = l_min

    elif flag == "td":
        l_min = lamb[2]
        b_vec = np.sum(inp_data * np.tile(lmn[:, 2], (n_t, 1)), axis=1)

        inp_data_2 = inp_data.copy()
        inp_data_2 -= np.tile(b_vec, (3, 1)).T * np.tile(lmn[:, 2], (n_t, 1))

        lamb, lmn = _variance_frame(inp_data_2)
        lamb[2] = l_min

    # Keep the attributes of the input, in the LMN coordinate system
    out = new_xyz(inp, lmn, "lmn")

    return out, lamb, lmn
