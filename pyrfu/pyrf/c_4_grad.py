#!/usr/bin/env python

# Built-in imports
from typing import Any, Optional, Sequence

# 3rd party imports
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray import DataArray

# Local imports
from pyrfu.pyrf._units import _attrs, _power, _ratio, _units
from pyrfu.pyrf.avg_4sc import avg_4sc
from pyrfu.pyrf.c_4_k import c_4_k
from pyrfu.pyrf.normalize import normalize

from .resample import resample


def _to_ts(out_data: NDArray[Any], time: DataArray) -> DataArray:
    r"""Converts output data to xarray.DataArray.

    Parameters
    ----------
    out_data : numpy.ndarray
        Output data.
    time : xarray.DataArray
        Time coordinate of the field.

    Returns
    -------
    DataArray
        Time series of the input field.

    """
    if out_data.ndim == 1:
        out = xr.DataArray(out_data, coords=[time], dims=["time"])

    elif out_data.ndim == 2:
        out = xr.DataArray(
            out_data,
            coords=[time, ["x", "y", "z"]],
            dims=["time", "comp"],
        )

    else:
        out = xr.DataArray(
            out_data,
            coords=[time, ["x", "y", "z"], ["x", "y", "z"]],
            dims=["time", "vcomp", "hcomp"],
        )

    return out


def _derivative(
    k_s: NDArray[Any],
    b_s: NDArray[Any],
    method: str,
    b_avg: Optional[NDArray[Any]] = None,
) -> NDArray[Any]:
    r"""Derivative `method` ("grad", "div", "curl" or "bdivb") of a field,
    from the reciprocal vectors `k_s` (spacecraft, time, 3) and the field
    `b_s` (spacecraft, time, ...) of the four spacecraft on the same times;
    `b_avg` (time, 3) is the field at the center of mass for "bdivb". The
    sums over the spacecraft are in the order of the loops they replaced.
    Shared with c_4_j, which computes three derivatives with one resampling
    and one c_4_k."""
    # Divergence of a tensor: d T_ij / d x_i
    if b_s.ndim == 4:
        return np.sum(np.einsum("sti,stij->stj", k_s, b_s), axis=0)

    if method == "div":
        return np.sum(np.sum(k_s * b_s, axis=2), axis=0)

    if method == "curl":
        return np.sum(np.cross(k_s, b_s, axis=2), axis=0)

    # Gradient of a scalar (time, i) or of a vector (time, j, i) = d b_j / d x_i
    # (for "grad" and "bdivb" only; it was computed for every method)
    if b_s.ndim == 2:
        grad_b = np.sum(k_s * b_s[:, :, np.newaxis], axis=0)
    else:
        grad_b = np.sum(k_s[:, :, np.newaxis, :] * b_s[:, :, :, np.newaxis], axis=0)

    if method == "grad":
        return grad_b

    # (B.grad)B
    return np.sum(b_avg[:, np.newaxis, :] * grad_b, axis=2)


def c_4_grad(
    r_list: Sequence[DataArray],
    b_list: Sequence[DataArray],
    method: Optional[str] = "grad",
) -> DataArray:
    r"""Calculate gradient of physical field using 4 spacecraft technique in [2]_ [3]_.

    Parameters
    ----------
    r_list : list of DataArray
        Time series of the positions of the spacecraft
    b_list : list of DataArray
        Time series of the magnetic field at the corresponding positions, or
        of any scalar or vector field, or of a tensor field (time, 3, 3) for
        "div" only.
    method : {"grad", "div", "curl", "bdivb", "curv"}, Optional
        Method flag :
            * "grad" : compute gradient (default), d B_j / d x_i at
              [time, j, i] for a vector field
            * "div" : compute divergence; for a tensor field T, the vector
              d T_ij / d x_i (summed over i, the divergence of each column),
              e.g. of a pressure tensor
            * "curl" : compute curl
            * "bdivb" : compute (B.grad) B, with B the average of the four
              fields
            * "curv" : compute curvature (b.grad) b of the unit vectors
              b = B / abs(B), as "bdivb" of the four b (see Notes)

    Returns
    -------
    DataArray
        Time series of the derivative of the input field corresponding to the method

    Raises
    ------
    TypeError
        If r_list or b_list is not a list or tuple of DataArray.
    ValueError
        If r_list or b_list doesn't have 4 elements, if method is not one of
        the above, or if the field is a tensor and method is not "div" (all
        the methods raised a broadcasting ValueError for a tensor), or if
        the field is a scalar and method is not "grad" (unrelated errors, or
        numbers for "div" with 3 samples).

    Notes
    -----
    The derivatives are those of a field varying linearly over the
    tetrahedron. For "curv", the b in front of the gradient is the average
    of the four unit vectors, which is not normalised (as irfu-matlab
    c_4_grad): it is shorter than one by a term of order (L / R_c)^2, with L
    the size of the tetrahedron and R_c the radius of curvature, and this
    partly compensates the error of the linear estimate. On circular field
    lines with the spacecraft at L = 0.3 R_c from the barycentre, the
    curvature is 0.997 / R_c (1.014 / R_c with the average normalised).

    See also
    --------
    pyrfu.pyrf.c_4_k : Calculates reciprocal vectors in barycentric coordinates.

    References
    ----------
    .. [2]	Dunlop, M. W., A. Balogh, K.-H. Glassmeier, and P. Robert (2002a),
            Four-point Cluster application of	magnetic field analysis
            tools: The Curl- ometer, J. Geophys. Res., 107(A11), 1384,
            doi : https://doi.org/10.1029/2001JA005088.

    .. [3]	Robert, P., et al. (1998), Accuracy of current determination, in
            Analysis Methods for Multi-Spacecraft Data, edited by G.
            Paschmann and P. W. Daly, pp. 395–418, Int. Space Sci. Inst.,
            Bern. doi : https://www.issibern.ch/forads/sr-001-16.pdf

    Examples
    --------
    >>> from pyrfu.mms import get_data
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Load magnetic field and spacecraft position

    >>> b_mms = [get_data("B_gse_fgm_srvy_l2", tint, m) for m in range(1, 5)]
    >>> r_mms = [get_data("R_gse", tint, m) for m in range(1, 5)]
    >>> gradb = pyrf.c_4_grad(r_mms, b_mms, "grad")

    """

    if not isinstance(r_list, (list, tuple)) or not isinstance(b_list, (list, tuple)):
        raise TypeError("r_list and b_list must be lists or tuples")

    if len(r_list) != 4 or len(b_list) != 4:
        raise ValueError("r_list and b_list must have 4 elements")

    if not all(isinstance(r, DataArray) for r in r_list):
        raise TypeError("All elements of r_list must be xarray.DataArray")

    if not all(isinstance(b, DataArray) for b in b_list):
        raise TypeError("All elements of b_list must be xarray.DataArray")

    if method not in ["grad", "div", "curl", "bdivb", "curv"]:
        raise ValueError(
            "Invalid method. Must be one of 'grad', 'div', 'curl', 'bdivb', 'curv'"
        )

    # A tensor field has only its divergence
    is_tensor = b_list[0].ndim == 3

    # Units: those of the field per those of the positions, the field squared
    # for (B.grad)B, per those of the positions for the curvature (none)
    units_b, units_r = _units(b_list[0]), _units(r_list[0])
    units = {
        "bdivb": _ratio(_power(units_b, 2), units_r),
        "curv": _ratio("", units_r),
    }.get(method, _ratio(units_b, units_r))

    if is_tensor and method != "div":
        raise ValueError(
            f"a tensor field has only the divergence (method 'div'), not '{method}'"
        )

    if b_list[0].ndim == 1 and method != "grad":
        raise ValueError(
            f"a scalar field has only the gradient (method 'grad'), not '{method}'"
        )

    # Resample with respect to 1st spacecraft
    r_list = [resample(r, b_list[0]) for r in r_list]
    b_list = [resample(b, b_list[0]) for b in b_list]

    # Reciprocal vectors in barycentric coordinates (see c_4_k) and the field
    # of the four spacecraft, (spacecraft, time, ...): the sums over the
    # spacecraft are in the same order as the loops they replace
    k_s = np.stack([k.data for k in c_4_k(r_list)])
    b_s = np.stack([b.data for b in b_list])
    time = b_list[0].time

    if method == "bdivb":
        # (B.grad)B with B at the center of mass of the tetrahedron (computed
        # for this method only; it was on every call)
        out_data = _derivative(k_s, b_s, method, avg_4sc(b_list).data)
    elif method == "curv":
        # Curvature: (b.grad)b of the unit vectors, with the same reciprocal
        # vectors (c_4_grad of the unit vectors resampled the positions and
        # computed them again)
        b_hat_list = [normalize(b) for b in b_list]
        b_hat_s = np.stack([b.data for b in b_hat_list])
        out_data = _derivative(k_s, b_hat_s, "bdivb", avg_4sc(b_hat_list).data)
    else:
        out_data = _derivative(k_s, b_s, method)

    out = _to_ts(out_data, time)
    out.attrs = _attrs({}, units)

    return out
