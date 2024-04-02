#!/usr/bin/env python

# Built-in imports
import itertools
from typing import Any, Mapping, Optional, Sequence

# 3rd party imports
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray.core.dataarray import DataArray

from pyrfu.pyrf.avg_4sc import avg_4sc
from pyrfu.pyrf.c_4_k import c_4_k
from pyrfu.pyrf.cross import cross
from pyrfu.pyrf.dot import dot
from pyrfu.pyrf.normalize import normalize

# Local imports
from .resample import resample

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


def _to_ts(out_data: NDArray[Any], b_dict: Mapping[str, DataArray]) -> DataArray:
    r"""Converts output data to xarray.DataArray.

    Parameters
    ----------
    out_data : numpy.ndarray
        Output data.
    b_dict : dict
        Dictionary containing the coordinates of the field.

    Returns
    -------
    DataArray
        Time series of the input field.

    """
    if out_data.ndim == 1:
        out = xr.DataArray(out_data, coords=[b_dict["1"].time], dims=["time"])

    elif out_data.ndim == 2:
        out = xr.DataArray(
            out_data,
            coords=[b_dict["1"].time, ["x", "y", "z"]],
            dims=["time", "comp"],
        )

    else:
        out = xr.DataArray(
            out_data,
            coords=[b_dict["1"].time, ["x", "y", "z"], ["x", "y", "z"]],
            dims=["time", "vcomp", "hcomp"],
        )

    return out


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
        Time series of the magnetic field at the corresponding positions
    method : {"grad", "div", "curl", "bdivb", "curv"}, Optional
        Method flag :
            * "grad" : compute gradient (default)
            * "div" : compute divergence
            * "curl" : compute curl
            * "bdivb" : compute b.div(b)
            * "curv" : compute curvature

    Returns
    -------
    DataArray
        Time series of the derivative of the input field corresponding to the method

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

    assert isinstance(r_list, list) and len(r_list) == 4, "r_list must a list of s/c"
    assert isinstance(b_list, list) and len(b_list) == 4, "b_list must a list of s/c"

    assert isinstance(method, str), "method must be a string"
    assert method.lower() in ["grad", "div", "curl", "bdivb", "curv"], "Invalid method"

    # Resample with respect to 1st spacecraft
    r_list = [resample(r, b_list[0]) for r in r_list]
    b_list = [resample(b, b_list[0]) for b in b_list]

    # Compute reciprocal vectors in barycentric coordinates (see c_4_k)
    k_list = c_4_k(r_list)

    # Magnetic field at the center of mass of the tetrahedron
    b_avg = avg_4sc(b_list)

    b_dict = {"1": b_list[0], "2": b_list[1], "3": b_list[2], "4": b_list[3]}
    k_dict = {"1": k_list[0], "2": k_list[1], "3": k_list[2], "4": k_list[3]}

    mms_list = b_dict.keys()

    # Gradient of scalar/vector
    if len(b_dict["1"].shape) == 1:
        grad_b = np.zeros((len(b_dict["1"]), 3))

        for i_sc in mms_list:
            grad_b += k_dict[i_sc].data * np.tile(b_dict[i_sc].data, (3, 1)).T

    else:
        grad_b = np.zeros((len(b_dict["1"]), 3, 3))

        for i, j, i_sc in itertools.product(range(3), range(3), mms_list):
            grad_b[:, j, i] += k_dict[i_sc][:, i].data * b_dict[i_sc][:, j].data

    # Gradient
    if method.lower() == "grad":
        out_data = grad_b

    # Divergence
    elif method.lower() == "div":
        out_data = np.zeros(len(b_dict["1"]))

        for mms_id in mms_list:
            out_data += dot(k_dict[mms_id], b_dict[mms_id]).data

    # Curl
    elif method.lower() == "curl":
        out_data = np.zeros((len(b_dict["1"]), 3))

        for mms_id in mms_list:
            out_data += cross(k_dict[mms_id], b_dict[mms_id]).data

    # B.div(B)
    elif method.lower() == "bdivb":
        out_data = np.zeros(b_avg.shape)

        for i in range(3):
            out_data[:, i] = np.sum(b_avg.data * grad_b[:, i, :], axis=1)

    # Curvature
    else:
        b_hat_list = [normalize(b) for b in b_list]

        out_data = c_4_grad(r_list, b_hat_list, method="bdivb").data

    out = _to_ts(out_data, b_dict)

    return out
