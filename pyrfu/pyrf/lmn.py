#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np

# Local imports
from .resample import resample
from .ts_vec_xyz import ts_vec_xyz


def _unit(vec):
    return vec / np.linalg.norm(vec, axis=-1, keepdims=True)


def lmn(inp, b_xyz, n, flag: str = "L"):
    r"""Field in the LMN frame defined by the magnetic field and a normal, as
    irfu-matlab ``irf_lmn``.

    Parameters
    ----------
    inp : xarray.DataArray
        Time series of the vector to transform.
    b_xyz : xarray.DataArray
        Time series of the magnetic field, in the coordinate system of `inp`.
    n : array_like
        Normal direction (3 values), e.g. of the current sheet or of the
        magnetopause, in the coordinate system of `inp`.
    flag : {"L", "N"}, Optional
        "L" (default): L along B at each time (B resampled to the times of
        `inp`), N along the component of n perpendicular to B, M = N x L; the
        frame follows B. "N": N along n, L along the component of the mean of
        B (over all its times) perpendicular to n, M = N x L.

    Returns
    -------
    out : xarray.DataArray
        Time series of `inp` in the LMN frame (L, M, N), with the attributes
        of `inp` and ``COORDINATE_SYSTEM`` "lmn".

    Raises
    ------
    ValueError
        If `n` is not 3 values or `flag` is not "L" or "N".

    Notes
    -----
    The frame is right-handed (L x M = N). With ``flag="N"``, L is along the
    mean B, as the documentation of irfu-matlab ``irf_lmn`` and
    :func:`pyrfu.pyrf.eb_nrf` with ``"b"``: irfu-matlab gives L and M the
    opposite sign (M = <B> x N since 2012). A frame given by its L, M and N
    vectors is :func:`pyrfu.pyrf.new_xyz`.

    See Also
    --------
    pyrfu.pyrf.new_xyz : Transform to a given frame.
    pyrfu.pyrf.mva : Minimum variance analysis.
    pyrfu.pyrf.eb_nrf : Electric field in the LMN frame of a velocity.

    Examples
    --------
    >>> from pyrfu import mms, pyrf

    Time interval

    >>> tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Magnetic field and electric field

    >>> b_xyz = mms.get_data("b_gse_fgm_srvy_l2", tint, 1)
    >>> e_xyz = mms.get_data("e_gse_edp_fast_l2", tint, 1)

    Electric field in the LMN frame of the mean B and of a normal along Z

    >>> e_lmn = pyrf.lmn(e_xyz, b_xyz, [0.0, 0.0, 1.0], "N")

    """

    n = np.asarray(n, dtype=np.float64)

    if n.shape != (3,):
        raise ValueError("n must be a vector (3 values)")

    if flag == "L":
        b_data = resample(b_xyz, inp).data
        n_l = _unit(b_data)
        n_n = _unit(np.cross(b_data, np.cross(n, b_data)))
        n_m = np.cross(n_n, n_l)
    elif flag == "N":
        n_n = _unit(n)
        n_m = _unit(np.cross(n_n, np.nanmean(b_xyz.data, axis=0)))
        n_l = np.cross(n_m, n_n)
    else:
        raise ValueError(f'flag must be "L" or "N", not {flag!r}')

    out_data = np.stack(
        [np.sum(inp.data * vec, axis=-1) for vec in [n_l, n_m, n_n]], axis=1
    )

    return ts_vec_xyz(
        inp.time.data, out_data, {**inp.attrs, "COORDINATE_SYSTEM": "lmn"}
    )
