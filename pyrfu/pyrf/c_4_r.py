#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from pyrfu._core.units import _attrs, _power, _units

from .resample import resample
from .ts_scalar import ts_scalar
from .ts_tensor_xyz import ts_tensor_xyz


def c_4_r(r_list):
    r"""Volumetric tensor of the tetrahedron of the four spacecraft, and its
    characteristic size, elongation and planarity, as irfu-matlab ``c_4_r``.

    .. math::

        R_{jk} = \frac{1}{4} \sum_{\alpha=1}^{4} (r_{\alpha j} - r_{c j})
        (r_{\alpha k} - r_{c k}), \quad L = 2a, \quad E = 1 - \frac{b}{a},
        \quad P = 1 - \frac{c}{b}

    with :math:`\mathbf{r}_c` the barycentre and :math:`a \geq b \geq c` the
    square roots of the eigenvalues of R, the semi-axes of the ellipsoid of
    the tetrahedron [1]_ [2]_.

    Parameters
    ----------
    r_list : list of xarray.DataArray
        Time series of the positions of the four spacecraft, resampled to the
        times of the first if their times differ.

    Returns
    -------
    r_tensor : xarray.DataArray
        Time series of the volumetric tensor R, in the units of the positions
        squared (e.g. km^2).
    size : xarray.DataArray
        Time series of the characteristic size L, in the units of the
        positions.
    elongation : xarray.DataArray
        Time series of the elongation E, from 0 (b = a) to 1 (a line).
    planarity : xarray.DataArray
        Time series of the planarity P, from 0 (c = b) to 1 (a plane).

    Raises
    ------
    ValueError
        If `r_list` does not have 4 elements.

    Notes
    -----
    A regular tetrahedron has E = P = 0. The planarity is NaN for spacecraft
    on a line (b = 0). The barycentre is :func:`pyrfu.pyrf.avg_4sc` of the
    positions, and the inverse of R is ``np.linalg.inv(r_tensor.data)``.

    References
    ----------
    .. [1]  Harvey, C. C. (1998), Spatial gradients and the volumetric tensor,
            in Analysis Methods for Multi-Spacecraft Data, edited by G.
            Paschmann and P. W. Daly, ISSI SR-001, pp. 307-322, Int. Space
            Sci. Inst., Bern.

    .. [2]  Robert, P., Roux, A., Harvey, C. C., Dunlop, M. W., Daly, P. W.,
            and Glassmeier, K.-H. (1998), Tetrahedron geometric factors, in
            Analysis Methods for Multi-Spacecraft Data, edited by G. Paschmann
            and P. W. Daly, ISSI SR-001, pp. 323-348, Int. Space Sci. Inst.,
            Bern.

    See Also
    --------
    pyrfu.pyrf.c_4_k : Reciprocal vectors.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Positions of the four spacecraft

    >>> r_mms = mms.get_data("r_gse_mec_srvy_l2", tint, [1, 2, 3, 4])

    Volumetric tensor, size, elongation and planarity of the tetrahedron

    >>> r_tensor, size, elongation, planarity = pyrf.c_4_r(r_mms)

    """

    if len(r_list) != 4:
        raise ValueError("r_list must contain exactly 4 elements.")

    time = r_list[0].time.data
    r_data = np.stack(
        [
            r.data if np.array_equal(r.time.data, time) else resample(r, r_list[0]).data
            for r in r_list
        ]
    )

    # Positions relative to the barycentre, and the tensor
    d_r = r_data - np.mean(r_data, axis=0)
    r_tensor = np.einsum("sti,stj->tij", d_r, d_r) / 4

    # Semi-axes a >= b >= c (rounding can make the smallest eigenvalue of a
    # planar tetrahedron slightly negative); none where a position is NaN
    finite = np.all(np.isfinite(r_tensor), axis=(1, 2))
    axes = np.full((len(time), 3), np.nan)
    eig = np.linalg.eigvalsh(r_tensor[finite])
    axes[finite] = np.sqrt(np.clip(eig, 0.0, None))[:, ::-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        elongation = 1.0 - axes[:, 1] / axes[:, 0]
        planarity = 1.0 - axes[:, 2] / axes[:, 1]

    units = _units(r_list[0])
    r_tensor = ts_tensor_xyz(time, r_tensor, _attrs({}, _power(units, 2)))
    size = ts_scalar(time, 2.0 * axes[:, 0], _attrs({}, units))
    elongation = ts_scalar(time, elongation, {"UNITS": ""})
    planarity = ts_scalar(time, planarity, {"UNITS": ""})

    return r_tensor, size, elongation, planarity
