#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os

# 3rd party imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import matplotlib.dates as mdates

# Local imports
from ..mms import get_data, get_feeps_omni
from ..pyrf import (iso86012datetime64, datetime642iso8601,
                    iso86012datetime, date_str)

from .plot_line import plot_line
from .plot_spectr import plot_spectr
from .plot_magnetosphere import plot_magnetosphere
from .span_tint import span_tint

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _tcut_edges(tint):
    tint = iso86012datetime64(np.array(tint))
    tint[0] += np.timedelta64(1, "s")
    tint[1] -= np.timedelta64(1, "s")
    tint = list(datetime642iso8601(tint))
    return tint


def _get_sclocs(data_rate, tint, mms_id, data_path):
    r_xyz = get_data(f"r_gse_mec_{data_rate}_l2", tint, mms_id,
                     data_path=data_path)
    return r_xyz


def _get_fields(data_rate, tint, mms_id, data_path):
    if data_rate == "fast":
        data_ratb = "srvy"
    else:
        data_ratb = data_rate

    b_xyz = get_data(f"b_gse_fgm_{data_ratb}_l2", tint, mms_id,
                     data_path=data_path)
    e_xyz = get_data(f"e_gse_edp_{data_rate}_l2", tint, mms_id,
                     data_path=data_path)
    return b_xyz, e_xyz


def _get_momnts(data_rate, tint, mms_id, data_path):
    n_i = get_data(f"ni_fpi_{data_rate}_l2", tint, mms_id,
                   data_path=data_path)
    n_e = get_data(f"ne_fpi_{data_rate}_l2", tint, mms_id,
                   data_path=data_path)

    v_xyz_i = get_data(f"vi_gse_fpi_{data_rate}_l2", tint, mms_id,
                       data_path=data_path)
    v_xyz_e = get_data(f"ve_gse_fpi_{data_rate}_l2", tint, mms_id,
                       data_path=data_path)
    return n_i, n_e, v_xyz_i, v_xyz_e


def _get_spectr(data_rate, tint, mms_id, data_path):
    if data_rate == "fast":
        data_ratp = "srvy"
    else:
        data_ratp = data_rate

    # FPI-DIS and FPI-DES differential energy flux
    def_i = get_data(f"defi_fpi_{data_rate}_l2", tint, mms_id,
                     data_path=data_path)
    def_e = get_data(f"defe_fpi_{data_rate}_l2", tint, mms_id,
                     data_path=data_path)

    if data_rate == "brst":
        dpf_i = get_feeps_omni(f"fluxi_{data_ratp}_l2", tint, mms_id,
                               data_path=data_path)
        dpf_e = get_feeps_omni(f"fluxe_{data_ratp}_l2", tint, mms_id,
                               data_path=data_path)
    else:
        dpf_i, dpf_e = [None, None]

    return def_i, def_e, dpf_i, dpf_e


def _init_fig():
    fig = plt.figure(figsize=(16 * 1.2, 9 * 1.2))
    gsp1 = fig.add_gridspec(18, 4, top=.95, bottom=.05, left=.08, right=.92,
                            wspace=1, hspace=0.1)

    gsp10 = gsp1[:5, :2].subgridspec(1, 2, hspace=0)
    gsp11 = gsp1[6:, :2].subgridspec(6, 1, hspace=0)
    gsp20 = gsp1[:, 2:].subgridspec(9, 1, hspace=0)

    # Create axes in the grid spec
    axs10 = [fig.add_subplot(gsp10[i]) for i in range(2)]
    axs11 = [fig.add_subplot(gsp11[i]) for i in range(6)]
    axs20 = [fig.add_subplot(gsp20[i]) for i in range(9)]

    return fig, axs10, axs11, axs20


def _plot_scps(axs, r_xyz):
    tint = list(datetime642iso8601(r_xyz.time.data[[0, -1]]))
    r_avg = np.mean(r_xyz.data / 6371, axis=0)

    field_lines = [False, False]
    for i, y_axis in enumerate(["$Y$ [$R_E$]", "$Z$ [$R_E$]"]):
        plot_magnetosphere(axs[i], tint, field_lines=field_lines[i])
        axs[i].invert_xaxis()
        axs[i].plot(r_avg[0], r_avg[i + 1], marker="^", color="tab:red",
                    linestyle="", label="MMS")
        axs[i].set_xlim([-30, 25])
        axs[i].set_ylim([-25, 25])
        axs[i].set_aspect("equal")
        axs[i].set_xlabel("$X$ [$R_E$]")
        axs[i].set_ylabel(y_axis)
        axs[i].invert_xaxis()

    return axs


def _plot_fast(axs, fields, momnts, spectr):
    b_xyz, _ = fields
    n_i, n_e, v_xyz_i, v_xyz_e = momnts
    def_i, def_e, _, _ = spectr

    plot_line(axs[0], b_xyz)
    axs[0].set_ylabel("$B$" + "\n" + "[nT]")
    axs[0].legend(["$B_x$", "$B_y$", "$B_z$"], frameon=False,
                  loc="upper left", bbox_to_anchor=(1., 1.))

    plot_line(axs[1], n_i)
    plot_line(axs[1], n_e)
    axs[1].set_ylabel("$n$" + "\n" + "[cm$^{-3}$]")
    axs[1].legend(["$n_{i}$", "$n_{e}$"], frameon=False,
                  loc="upper left", bbox_to_anchor=(1., 1.))

    plot_line(axs[2], v_xyz_i)
    axs[2].set_ylabel("$V_i$" + "\n" + "[km s$^{-1}$]")
    axs[2].legend(["$V_{i,x}$", "$V_{i,y}$", "$V_{i,z}$"], frameon=False,
                  loc="upper left", bbox_to_anchor=(1., 1.))

    axs[3], caxs3 = plot_spectr(axs[3], def_i, yscale="log", cscale="log")
    axs[3].set_ylabel("$E_i$" + "\n" + "[eV]")
    caxs3.set_ylabel("DEF" + "\n" + "[(cm$^2$ s sr)$^{-1}$]")

    plot_line(axs[4], v_xyz_e)
    axs[4].set_ylabel("$V_e$" + "\n" + "[km s$^{-1}$]")
    axs[4].legend(["$V_{e,x}$", "$V_{e,y}$", "$V_{e,z}$"], frameon=False,
                  loc="upper left", bbox_to_anchor=(1., 1.))

    axs[5], caxs5 = plot_spectr(axs[5], def_e, yscale="log", cscale="log")
    axs[5].set_ylabel("$E_e$" + "\n" + "[eV]")
    caxs5.set_ylabel("DEF" + "\n" + "[(cm$^2$ s sr)$^{-1}$]")

    axs[-1].get_shared_x_axes().join(*axs)

    for ax in axs[:-1]:
        ax.xaxis.set_ticklabels([])

    return axs


def _plot_brst(axs, fields, momnts, spectr):
    b_xyz, e_xyz = fields
    n_i, n_e, v_xyz_i, v_xyz_e = momnts
    def_i, def_e, dpf_i, dpf_e = spectr

    plot_line(axs[0], b_xyz)
    axs[0].set_ylabel("$B$" + "\n" + "[nT]")
    axs[0].legend(["$B_x$", "$B_y$", "$B_z$"], frameon=False,
                  loc="upper left", bbox_to_anchor=(1., 1.))

    plot_line(axs[1], n_i)
    plot_line(axs[1], n_e)
    axs[1].set_ylabel("$n$" + "\n" + "[cm$^{-3}$]")
    axs[1].legend(["$n_{i}$", "$n_{e}$"], frameon=False,
                  loc="upper left", bbox_to_anchor=(1., 1.))

    plot_line(axs[2], v_xyz_i)
    axs[2].set_ylabel("$V_i$" + "\n" + "[km s$^{-1}$]")
    axs[2].legend(["$V_{i,x}$", "$V_{i,y}$", "$V_{i,z}$"], frameon=False,
                  loc="upper left", bbox_to_anchor=(1., 1.))

    axs[3], caxs3 = plot_spectr(axs[3], dpf_i[:, 1:], yscale="log",
                                cscale="log")
    axs[3].set_ylabel("$E_i$" + "\n" + "[keV]")
    caxs3.set_ylabel("Intensity" + "\n" + "[(cm$^2$ s sr keV)$^{-1}$]")

    axs[4], caxs4 = plot_spectr(axs[4], def_i, yscale="log", cscale="log")
    axs[4].set_ylabel("$E_i$" + "\n" + "[eV]")
    caxs4.set_ylabel("DEF" + "\n" + "[(cm$^2$ s sr)$^{-1}$]")

    plot_line(axs[5], v_xyz_e)
    axs[5].set_ylabel("$V_e$" + "\n" + "[km s$^{-1}$]")
    axs[5].legend(["$V_{e,x}$", "$V_{e,y}$", "$V_{e,z}$"], frameon=False,
                  loc="upper left", bbox_to_anchor=(1., 1.))

    axs[6], caxs6 = plot_spectr(axs[6], dpf_e[:, 1:], yscale="log",
                                cscale="log")
    axs[6].set_ylabel("$E_e$" + "\n" + "[keV]")
    caxs6.set_ylabel("Intensity" + "\n" + "[(cm$^2$ s sr keV)$^{-1}$]")

    axs[7], caxs7 = plot_spectr(axs[7], def_e, yscale="log", cscale="log")
    axs[7].set_ylabel("$E_e$" + "\n" + "[eV]")
    caxs7.set_ylabel("DEF" + "\n" + "[(cm$^2$ s sr)$^{-1}$]")

    plot_line(axs[8], e_xyz)
    axs[8].set_ylabel("$E$" + "\n" + "[mV m$^{-1}$]")
    axs[8].legend(["$E_x$", "$E_y$", "$E_z$"], frameon=False,
                  loc="upper left", bbox_to_anchor=(1., 1.))

    axs[-1].get_shared_x_axes().join(*axs)

    for ax in axs[:-1]:
        ax.xaxis.set_ticklabels([])

    return axs


def _add_logo(fig, path, loc=None):
    if loc is None:
        loc = [-0.015, 0.885, 0.1, 0.1]
    im = mimg.imread(path)
    # put a new axes where you want the image to appear
    # (x, y, width, height)
    imax = fig.add_axes(loc)
    # remove ticks & the box from imax 
    imax.set_axis_off()
    # print the logo with aspect="equal" to avoid distorting the logo
    imax.imshow(im, aspect="equal")


def plot_sitl_overview(tint_brst, title, mms_id: int = 2,
                       data_path: str = "/Volumes/mms",
                       fig_path: str = "figures"):
    r"""Creates overview plot from SITL selections.

    Paramters
    ---------
    tint_brst : list
        Time interval selected.
    mms_id : int, Optional
        Spacecraft index. Default is 1.
    data_path : str, Optional
        Path to MMS data. Default is /Volumes/

    Returns
    -------
    fig : matplotlib.figure
        Figure.
    axs : list
        All axes.

    """

    file_name = "IRF_logo_blue_on_white.jpg"
    logo_path = os.sep.join([os.path.dirname(os.path.abspath(__file__)),
                            "logo", file_name])

    tint_dt = iso86012datetime(np.array(tint_brst))
    t_start = np.datetime64(
        f"{tint_brst[0][:11]}{tint_dt[0].hour - tint_dt[0].hour % 2:02}:00:00")

    if t_start + np.timedelta64(2, "h") < np.datetime64(tint_brst[1]):
        tint_tmp0 = np.array([tint_brst[0], t_start + np.timedelta64(2, "h")])
        tint_tmp1 = np.array([tint_tmp0[1], np.datetime64(tint_brst[1])])
        plot_sitl_overview(list(datetime642iso8601(tint_tmp0)), title, mms_id,
                           data_path)
        plot_sitl_overview(list(datetime642iso8601(tint_tmp1)), title, mms_id,
                           data_path)
    else:
        tint_fast = np.array([t_start, t_start + np.timedelta64(2, "h")])
        tint_fast = list(datetime642iso8601(tint_fast))
        tint_fast = _tcut_edges(tint_fast)
        tint_brst = _tcut_edges(tint_brst)

        r_xyz = _get_sclocs("srvy", tint_fast, mms_id, data_path)

        fields_fast = _get_fields("fast", tint_fast, mms_id, data_path)
        fields_brst = _get_fields("brst", tint_brst, mms_id, data_path)

        momnts_fast = _get_momnts("fast", tint_fast, mms_id, data_path)
        momnts_brst = _get_momnts("brst", tint_brst, mms_id, data_path)

        spectr_fast = _get_spectr("fast", tint_fast, mms_id, data_path)
        spectr_brst = _get_spectr("brst", tint_brst, mms_id, data_path)

        fig, axs10, axs11, axs20 = _init_fig()
        _ = _plot_scps(axs10, r_xyz)
        axs11 = _plot_fast(axs11, fields_fast, momnts_fast, spectr_fast)
        axs20 = _plot_brst(axs20, fields_brst, momnts_brst, spectr_brst)

        axs11[-1].set_xlim(mdates.datestr2num(tint_fast))
        axs20[-1].set_xlim(mdates.datestr2num(tint_brst))
        fig.align_ylabels(axs11)
        fig.align_ylabels(axs20)

        span_tint(axs11, tint_brst, ec="k", fc="tab:purple", alpha=.2)

        fig.suptitle(title)

        _add_logo(fig, logo_path)

        tint_iso = datetime642iso8601(iso86012datetime64(np.array(tint_brst)))
        pref = date_str([f"{t_[:-3]}" for t_ in list(tint_iso)], 4)
        fig.savefig(os.path.join(fig_path, f"{pref}_mms{mms_id}_overview.png"))

        # return fig, [*axs10, *axs11, *axs20]
        return
