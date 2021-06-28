#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from ..pyrf import resample

from .feeps_active_eyes import feeps_active_eyes

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"

# Rotation matrices for FEEPS coord system (FCS) into body coordinate system
# (BCS):
t_top = np.array([[1. / np.sqrt(2.), -1. / np.sqrt(2.), 0],
                  [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0], [0, 0, 1]])
t_bot = np.array([[-1. / np.sqrt(2.), -1. / np.sqrt(2.), 0],
                  [-1. / np.sqrt(2.), 1. / np.sqrt(2.), 0], [0, 0, -1]])

# the following 2 hash tables map TOP/BOTTOM telescope #s to index of the
# PA array created above
top_tele_idx_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 0, 7: 1, 8: 2, 9: 5,
                    10: 6, 11: 7, 12: 8}

bot_tele_idx_map = {1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 3, 7: 4, 8: 5,
                    9: 14, 10: 15, 11: 16, 12: 17}

# Telescope vectors in FCS:
v_fcs = {1: [0.347, -0.837, 0.423], 2: [0.347, -0.837, -0.423],
         3: [0.837, -0.347, 0.423], 4: [0.837, -0.347, -0.423],
         5: [-0.087, 0.000, 0.996], 6: [0.104, 0.180, 0.978],
         7: [0.654, -0.377, 0.656], 8: [0.654, -0.377, -0.656],
         9: [0.837, 0.347, 0.423], 10: [0.837, 0.347, -0.423],
         11: [0.347, 0.837, 0.423], 12: [0.347, 0.837, -0.423]}

sensor_ids = {"electron": {"top": [1, 2, 3, 4, 5, 9, 10, 11, 12],
                           "bot": [1, 2, 3, 4, 5, 9, 10, 11, 12]},
              "ion": {"top": [6, 7, 8], "bot": [6, 7, 8]}}


def _calc_pas(d_type, b_bcs):
    n_t = len(b_bcs.time)
    n_scopes = len(sensor_ids[d_type]["top"])

    vt_bcs = {i: -np.matmul(t_top, v_fcs[i]) for i in range(1, 13)}
    vb_bcs = {i: -np.matmul(t_bot, v_fcs[i]) for i in range(1, 13)}

    pas = np.empty([n_t, int(2 * n_scopes)])
    for i in range(int(2 * n_scopes)):
        if i < n_scopes:
            v_bcs = vt_bcs[sensor_ids[d_type]["top"][i % n_scopes]]
        else:
            v_bcs = vb_bcs[sensor_ids[d_type]["bot"][i % n_scopes]]

        v_dot_b = np.sum(v_bcs * b_bcs, axis=1)
        v_dot_b /= np.linalg.norm(v_bcs) * np.linalg.norm(b_bcs.data, axis=1)
        pas[:, i] = np.rad2deg(np.arccos(v_dot_b))

    return pas


def _calc_new_pas(pas, b_bcs, eyes):
    n_t = len(b_bcs.time.data)
    n_top = len(eyes["top"])
    n_bot = len(eyes["bottom"])

    top_idxs, bot_idxs = [[], []]

    new_pas = np.empty([n_t, n_top + n_bot])

    for j, top_eye in enumerate(eyes["top"]):
        new_pas[:, j] = pas[:, top_tele_idx_map[top_eye]]
        top_idxs.append(j)

    for j, bot_eye in enumerate(eyes["bottom"]):
        new_pas[:, j + n_top] = pas[:, bot_tele_idx_map[bot_eye]]
        bot_idxs.append(j + n_top)

    return new_pas, top_idxs, bot_idxs


def feeps_pitch_angles(inp_dataset, b_bcs):
    r"""Computes the FEEPS pitch angles for each telescope from magnetic field
    data.

    Parameters
    ----------
    inp_dataset : xarray.Dataset
        Dataset of the time series of the energy spectrum for each eye of
        FEEPS telescopes.
    b_bcs : xarray.DataArray
        Time series of the magnetic in spacecraft coordinates system.

    Returns
    -------
    out : xarray.DataArray
        Time series of the pitch angles.
    idx_maps : dict
        to fill.

    """

    # get the times from the currently loaded FEEPS data
    times = inp_dataset.time.data

    d_type = inp_dataset.attrs["dtype"]
    d_rate = inp_dataset.attrs["tmmode"]
    mms_id = inp_dataset.attrs["mmsId"]

    tint = np.datetime_as_string(np.hstack([np.min(times),
                                            np.max(times)]), "ns")

    eyes = feeps_active_eyes(inp_dataset.attrs, list(tint), mms_id)

    idx_maps = None

    # pitch angles for each eye at each time
    pas = _calc_pas(d_type, b_bcs)

    if d_type == "electron" and d_rate == "brst":
        new_pas = pas
    else:
        # PAs for only active eyes
        # pitch angles for each eye at eaceh time
        new_pas, top_idxs, bot_idxs = _calc_new_pas(pas, b_bcs, eyes)
        idx_maps = {f"{d_type}-top": top_idxs, f"{d_type}-bottom": bot_idxs}

    out = xr.DataArray(new_pas,
                       coords=[b_bcs.time.data, np.arange(new_pas.shape[1])],
                       dims=["time", "idx"])
    out = resample(out, inp_dataset.time)

    return out, idx_maps

