#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
from xarray import DataArray, Dataset


def _pitch_edges(pitch, attrs):
    r"""Lower and upper edges (deg) of the pitch-angle bins: the
    delta_pitchangle_minus/plus attributes, else half the spacing, within 0
    to 180 degrees."""

    if "delta_pitchangle_minus" in attrs and "delta_pitchangle_plus" in attrs:
        d_minus = np.asarray(attrs["delta_pitchangle_minus"], dtype=np.float64)
        d_plus = np.asarray(attrs["delta_pitchangle_plus"], dtype=np.float64)
    elif len(pitch) > 1:
        d_minus = d_plus = np.abs(np.median(np.diff(pitch))) / 2
    else:
        raise ValueError(
            "a single pitch angle needs delta_pitchangle_minus/plus attributes"
        )

    return np.clip(pitch - d_minus, 0.0, 180.0), np.clip(pitch + d_plus, 0.0, 180.0)


def pitch_angle_average(pad, pitch_angles=None):
    r"""Average of a pitch-angle distribution over a range of pitch angles,
    weighted by the solid angle of the bins, as irfu-matlab
    ``irf.pitch_angle_average``.

    .. math::

        \bar{f} = \frac{\sum_i f_i \, \Delta\Omega_i}{\sum_i \Delta\Omega_i},
        \quad \Delta\Omega_i = 2 \pi (\cos\alpha_i^- - \cos\alpha_i^+)

    over the bins i with data whose pitch angle is in the range, with
    :math:`\alpha_i^\pm` their edges.

    Parameters
    ----------
    pad : xarray.Dataset or xarray.DataArray
        Pitch-angle distribution: a Dataset (time, energy, pitch angle) as
        :func:`pyrfu.mms.get_pitch_angle_dist` or
        :func:`pyrfu.mms.edi_amb_pad`, or a DataArray with a ``theta``
        (pitch angle) dimension as :func:`pyrfu.mms.eis_pad`,
        :func:`pyrfu.mms.feeps_pad` or :func:`pyrfu.mms.hpca_pad`. The widths
        of the bins are the ``delta_pitchangle_minus/plus`` attributes, else
        half the spacing of the pitch angles.
    pitch_angles : array_like, Optional
        [min, max] (degrees): the bins whose pitch angle is in the range,
        bounds included. Default is all the bins.

    Returns
    -------
    out : xarray.Dataset or xarray.DataArray
        For a Dataset, a pitch-angle distribution of one bin spanning the
        selected bins (as :func:`pyrfu.mms.pad_palim` with a single angle),
        which the pitch-angle routines take; for a DataArray, `pad` without
        its ``theta`` dimension (e.g. time x energy). Both keep the units and
        attributes of `pad`.

    Raises
    ------
    TypeError
        If `pad` is neither a xarray.Dataset nor a xarray.DataArray.
    ValueError
        If `pad` has no pitch-angle dimension, `pitch_angles` is not [min,
        max], or no bin is in the range.

    Notes
    -----
    The NaN bins have no data and are left out, with their solid angles; a
    time (energy) without data in the range is NaN (see
    :ref:`api-mms-vdf-nan`). A bin counts whole if its pitch angle is in the
    range, as irfu-matlab, also when its edges are not.

    See Also
    --------
    pyrfu.mms.pad_palim : Select pitch angles.

    Examples
    --------
    >>> from pyrfu import mms, pyrf
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint, 1)
    >>> b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint, 1)
    >>> pad_e = mms.get_pitch_angle_dist(vdf_e, b_dmpa, angles=12)

    Field-aligned electrons, 0 to 30 degrees

    >>> pad_par = pyrf.pitch_angle_average(pad_e, [0.0, 30.0])

    """

    if isinstance(pad, Dataset):
        data = pad["data"]
        pitch = np.asarray(pad.theta.data, dtype=np.float64)
        pitch = pitch[0] if pitch.ndim == 2 else pitch
        axis = data.dims.index("idx1")
    elif isinstance(pad, DataArray):
        if "theta" not in pad.dims:
            raise ValueError("pad must have a theta (pitch angle) dimension")

        data = pad
        pitch = np.asarray(pad.theta.data, dtype=np.float64)
        axis = pad.dims.index("theta")
    else:
        raise TypeError("pad must be a xarray.Dataset or a xarray.DataArray")

    lower, upper = _pitch_edges(pitch, pad.attrs)
    lower = np.broadcast_to(lower, pitch.shape)
    upper = np.broadcast_to(upper, pitch.shape)

    if pitch_angles is None:
        pitch_angles = [np.min(pitch), np.max(pitch)]

    pitch_angles = np.asarray(pitch_angles, dtype=np.float64)

    if pitch_angles.shape != (2,):
        raise ValueError("pitch_angles must be [min, max]")

    index = np.where((pitch >= pitch_angles[0]) & (pitch <= pitch_angles[1]))[0]

    if index.size == 0:
        raise ValueError(f"no pitch-angle bin in {pitch_angles.tolist()}")

    # Solid angles of the selected bins, those with data only
    solid = np.cos(np.deg2rad(lower[index])) - np.cos(np.deg2rad(upper[index]))
    shape = [1] * data.ndim
    shape[axis] = index.size
    values = np.take(np.asarray(data.data, dtype=np.float64), index, axis=axis)
    weights = np.where(np.isfinite(values), solid.reshape(shape), 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        average = np.sum(np.nan_to_num(values) * weights, axis=axis, keepdims=True)
        average /= np.sum(weights, axis=axis, keepdims=True)

    if isinstance(pad, DataArray):
        out = pad.isel(theta=0, drop=True).copy(data=np.squeeze(average, axis=axis))
        out.attrs = dict(pad.attrs)
        return out

    # One bin spanning the selected bins
    centre = (lower[index[0]] + upper[index[-1]]) / 2
    out = pad.isel(idx1=index[:1])
    out["data"] = out.data.copy(data=average)
    out["theta"] = out.theta.copy(data=np.full(out.theta.shape, centre))
    out.attrs = {
        **pad.attrs,
        "delta_pitchangle_minus": np.array([centre - lower[index[0]]]),
        "delta_pitchangle_plus": np.array([upper[index[-1]] - centre]),
    }

    return out
