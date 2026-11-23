#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
from typing import Literal

# 3rd party imports
import numpy as np
from scipy import constants
from xarray import DataArray

# Local imports
from pyrfu._core.time import _seconds
from pyrfu._core.units import _check_units

from .filt import filt
from .resample import resample
from .ts_scalar import ts_scalar
from .ts_vec_xyz import ts_vec_xyz


def _unit(vec):
    with np.errstate(divide="ignore", invalid="ignore"):
        return vec / np.linalg.norm(vec, axis=1, keepdims=True)


def _on_times(inp, ref):
    r"""Values of a time series at the times of ref (resampled), or of a
    constant vector (3 values) repeated."""

    if isinstance(inp, DataArray):
        return resample(inp, ref).data

    inp = np.asarray(inp, dtype=np.float64)

    if inp.shape != (3,):
        raise ValueError("v_xyz must be a time series or a vector (3 values)")

    return np.tile(inp, (len(ref.time), 1))


def jz(
    v_xyz,
    b_xyz,
    db_xyz=None,
    n_av: int | None = None,
    method: Literal["A", "B"] = "B",
    deg_p: float = 10.0,
    deg_z: float = 10.0,
):
    r"""Current density from the time derivative of the magnetic field
    measured by one spacecraft crossing a current structure, as irfu-matlab
    ``irf_jz``.

    Method "B" neglects the changes of B along the velocity (all the current
    is perpendicular to it, e.g. a crossing of the magnetopause):

    .. math::

        \mathbf{J} = \frac{\mathbf{v} \times \partial_t \mathbf{B}}
        {\mu_0 v^2}

    Method "A" assumes plane current sheets (tangential discontinuities)
    whose normal may be at an angle to the velocity, e.g. auroral current
    sheets: the normal is along :math:`\partial_t \mathbf{B} \times
    \hat{\mathbf{b}}`, and the currents parallel and perpendicular to B are
    the changes of B across the sheet divided by the velocity along its
    normal, :math:`\mu_0 J_\parallel = |\partial_t \mathbf{B} \times
    \hat{\mathbf{b}}| / v_n` and :math:`\mu_0 J_\perp = |\partial_t
    \mathbf{B} \cdot \hat{\mathbf{b}}| / v_n`.

    Parameters
    ----------
    v_xyz : xarray.DataArray or array_like
        Time series or vector (3 values) of the velocity of the spacecraft
        relative to the structure [km/s], e.g. the negative of the plasma
        velocity when the structure moves with the plasma, in the coordinate
        system of `b_xyz`.
    b_xyz : xarray.DataArray
        Time series of the background magnetic field [nT].
    db_xyz : xarray.DataArray, Optional
        Time series of the disturbance magnetic field [nT], whose time
        derivative gives the current. Default is `b_xyz`.
    n_av : int, Optional
        Number of samples of `db_xyz` to average before the derivative: a
        low-pass filter at f_s / (2 `n_av`) (elliptic, order 3, as
        irfu-matlab ``irf_filt``), above 1. Default is None (no filter).
    method : {"A", "B"}, Optional
        "A": plane current sheets at an angle to the velocity; "B" (default):
        no change of B along the velocity.
    deg_p : float, Optional
        Method "A": the smallest angle (degrees) between the current sheet
        and the velocity perpendicular to B; NaN currents below. Default 10.
    deg_z : float, Optional
        Method "A": the smallest angle (degrees) between the velocity and B;
        NaN currents below. Default 10.

    Returns
    -------
    j_xyz : xarray.DataArray
        Method "B": time series of the current density [A/m^2].
    j_par : xarray.DataArray
        Time series of the current density along B [A/m^2]; method "A":
        positive along B.
    j_perp : xarray.DataArray
        Time series of the current density perpendicular to B [A/m^2]: along
        B x v for method "B"; for method "A", positive when the spacecraft
        crosses the sheet along its normal.
    n_j : xarray.DataArray
        Method "A": time series of the normal of the current sheet.
    angle : xarray.DataArray
        Method "A": time series of the angle (degrees) from the normal of the
        current sheet to the velocity perpendicular to B, anticlockwise about
        B (irfu-matlab: the azimuthal angle between the current sheet and
        the velocity perpendicular to B).

    The outputs are on the times of `db_xyz`: (j_xyz, j_par, j_perp) for
    method "B", (j_par, j_perp, n_j, angle) for method "A", as irfu-matlab.

    Raises
    ------
    ValueError
        If `method` is not "A" or "B", `n_av` is not above 1, or `v_xyz` is
        neither a time series nor 3 values.

    Notes
    -----
    Both methods assume a planar structure moving steadily past the
    spacecraft: across a three-dimensional or changing structure (e.g. the
    reconnection site of 2015-10-16 13:07, MMS1), the estimate does not
    reproduce the four-spacecraft current of `c_4_j` (correlations below
    0.5). The velocity is that of the spacecraft relative to the structure,
    not the plasma flow along it.

    The time derivative is the second-order central difference of
    ``np.gradient`` (one-sided at the ends), that of MATLAB ``gradient`` for
    uniform sampling. The constant :math:`\mu_0` is exact (irfu-matlab uses
    :math:`4 \pi \times 10^{-7}`, 1.3e-10 relative).

    See Also
    --------
    pyrfu.pyrf.c_4_j : Current density from four spacecraft.

    Examples
    --------
    >>> import numpy as np
    >>> from pyrfu import mms, pyrf
    >>> tint = ["2015-10-16T13:07:00.000", "2015-10-16T13:07:10.000"]
    >>> b_xyz = mms.get_data("b_gse_fgm_brst_l2", tint, 1)

    Velocity of the boundary along its normal, e.g. from the timing of its
    crossing by the four spacecraft (`pyrf.c_4_v`), here 30 km/s along -X

    >>> v_boundary = np.array([-30.0, 0.0, 0.0])
    >>> j_xyz, j_par, j_perp = pyrf.jz(-v_boundary, b_xyz)

    """

    if method not in ["A", "B"]:
        raise ValueError(f'method must be "A" or "B", not {method!r}')

    if n_av is not None and not n_av > 1:
        raise ValueError("n_av must be above 1")

    _check_units(b_xyz, "nT", "b_xyz")

    if isinstance(v_xyz, DataArray):
        _check_units(v_xyz, "km/s", "v_xyz")

    if db_xyz is None:
        db_xyz = b_xyz

    _check_units(db_xyz, "nT", "db_xyz")

    if n_av is not None:
        f_s = 1 / np.median(np.diff(_seconds(db_xyz.time.data, db_xyz.time.data[0])))
        db_xyz = filt(db_xyz, 0.0, f_s / (2 * n_av), 3)

    # Time derivative of the disturbance field, nT/s
    time = db_xyz.time.data
    dt_b = np.gradient(db_xyz.data, _seconds(time, time[0]), axis=0)

    v_data = _on_times(v_xyz, db_xyz)
    b_hat = _unit(resample(b_xyz, db_xyz).data)

    # nT/km to T/m
    factor = 1e-12 / constants.mu_0

    if method == "B":
        v_2 = np.sum(v_data**2, axis=1, keepdims=True)
        j_xyz = factor * np.cross(v_data, dt_b) / v_2
        j_par = np.sum(j_xyz * b_hat, axis=1)
        j_perp = np.sum(j_xyz * _unit(np.cross(b_hat, v_data)), axis=1)

        return (
            ts_vec_xyz(time, j_xyz, {"UNITS": "A/m^2"}),
            ts_scalar(time, j_par, {"UNITS": "A/m^2"}),
            ts_scalar(time, j_perp, {"UNITS": "A/m^2"}),
        )

    # Normal of the current sheet, and the velocities along it and
    # perpendicular to B
    dt_b_x_b = np.cross(dt_b, b_hat)
    dt_b_b = np.sum(dt_b * b_hat, axis=1)
    n_j = _unit(dt_b_x_b)
    v_n = np.sum(v_data * n_j, axis=1)
    v_perp = np.cross(b_hat, np.cross(v_data, b_hat))

    # Angle from the normal to the velocity perpendicular to B, about B
    sin_angle = np.sum(np.cross(_unit(v_perp), n_j) * b_hat, axis=1)
    sin_angle = np.where(np.sum(v_perp * n_j, axis=1) < 0, -sin_angle, sin_angle)
    angle = -np.rad2deg(np.arcsin(np.clip(sin_angle, -1.0, 1.0)))

    # Not valid for a velocity close to B or to the sheet
    abs_v_perp = np.linalg.norm(v_perp, axis=1)
    invalid = abs_v_perp < np.linalg.norm(v_data, axis=1) * np.sin(np.deg2rad(deg_z))
    invalid |= np.abs(v_n) < abs_v_perp * np.sin(np.deg2rad(deg_p))

    with np.errstate(divide="ignore", invalid="ignore"):
        j_par = factor * np.linalg.norm(dt_b_x_b, axis=1) / v_n
        j_perp = factor * np.abs(dt_b_b) / v_n

    j_par[invalid] = np.nan
    j_perp[invalid] = np.nan

    return (
        ts_scalar(time, j_par, {"UNITS": "A/m^2"}),
        ts_scalar(time, j_perp, {"UNITS": "A/m^2"}),
        ts_vec_xyz(time, n_j, {"UNITS": ""}),
        ts_scalar(time, angle, {"UNITS": "deg"}),
    )
