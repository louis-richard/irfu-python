#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np

# Local imports
from ._vdf_bins import _is_skymap, _pitch_angles, _polar_edges

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2026"
__license__ = "MIT"
__version__ = "2.6.0"
__status__ = "Prototype"


def pad_palim(pad, pitch_angles, average: bool = True):
    r"""Selects pitch angles of a pitch-angle distribution, as irfu-matlab
    PDist.palim.

    Parameters
    ----------
    pad : xarray.Dataset
        Pitch-angle distribution (time, energy, pitch angle), e.g. from
        `get_pitch_angle_dist`.
    pitch_angles : float or array_like
        [min, max]: the bins whose pitch angle is strictly between min and
        max (degrees). A single angle: the bin closest to it, or the bins
        equally close.
    average : bool, Optional
        With a single angle, average the equally close bins (NaN-mean) into
        one (True, default), or keep them (irfu-matlab ``"noav"``). Ignored
        for a range.

    Returns
    -------
    out : xarray.Dataset
        Pitch-angle distribution of the selected bins. An average bin is at
        the mean of their pitch angles and spans their widths
        (``delta_pitchangle_minus/plus``).

    Raises
    ------
    ValueError
        If `pad` is not a pitch-angle distribution or `pitch_angles` is not
        one or two angles.

    Examples
    --------
    >>> from pyrfu import mms
    >>> tint = ["2017-07-11T22:33:30.000", "2017-07-11T22:34:30.000"]
    >>> vdf_e = mms.get_data("pde_fpi_brst_l2", tint, 1)
    >>> b_dmpa = mms.get_data("b_dmpa_fgm_brst_l2", tint, 1)
    >>> pad_e = mms.get_pitch_angle_dist(vdf_e, b_dmpa, angles=12)
    >>> pad_perp = mms.pad_palim(pad_e, 90.0)

    """
    if _is_skymap(pad):
        raise ValueError("pad must be a pitch-angle distribution")

    pitch = _pitch_angles(pad)
    lower, upper = _polar_edges(pitch, pad.attrs, "pitchangle")
    lower, upper = np.broadcast_to(lower, pitch.shape), np.broadcast_to(
        upper, pitch.shape
    )
    pitch_angles = np.atleast_1d(np.asarray(pitch_angles, dtype=np.float64))

    if pitch_angles.size == 2:
        index = np.where((pitch > pitch_angles[0]) & (pitch < pitch_angles[1]))[0]
        average = False
    elif pitch_angles.size == 1:
        distance = np.abs(pitch - pitch_angles[0])
        index = np.where(distance == np.min(distance))[0]
    else:
        raise ValueError("pitch_angles must be one angle or [min, max]")

    out = pad.isel(idx1=index)
    out.attrs = {
        **pad.attrs,
        "delta_pitchangle_minus": pitch[index] - lower[index],
        "delta_pitchangle_plus": upper[index] - pitch[index],
    }

    if not average:
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data = np.nanmean(out.data.data, axis=2, keepdims=True)

    centre = np.mean(pitch[index])
    out = pad.isel(idx1=index[:1])
    out["data"] = out.data.copy(data=data)
    out["theta"] = out.theta.copy(data=np.full(out.theta.shape, centre))
    out.attrs = {
        **pad.attrs,
        "delta_pitchangle_minus": np.array([centre - lower[index[0]]]),
        "delta_pitchangle_plus": np.array([upper[index[-1]] - centre]),
    }
    return out
