#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
example_mms_ebfields.py

@author : Louis RICHARD
"""

import os
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pyrfu.mms import get_data
from pyrfu.pyrf import convert_fac, filt, wavelet, plasma_calc, date_str
from pyrfu.plot import plot_line, plot_spectr


def ccwt(s=None, nf=100, f=None, nc=100):
    """
    Compressed wavelet transform with average over nc time stamps
    """
    if f is None:
        f = [5e-1, 1e3]

    s_cwt = wavelet(s, nf=nf, f=f, plot=False)

    idx = np.arange(int(nc / 2), len(s_cwt.time) - int(nc / 2), step=nc).astype(int)
    s_cwt_times = s_cwt.time[idx]

    s_cwt_x, s_cwt_y, s_cwt_z = [np.zeros((len(idx), nf)) for _ in range(3)]

    for ii in range(len(idx)):
        s_cwt_x[ii, :] = np.squeeze(np.nanmean(s_cwt.x[idx[ii] - int(nc / 2) + 1:idx[ii] + int(nc / 2), :], axis=0))
        s_cwt_y[ii, :] = np.squeeze(np.nanmean(s_cwt.y[idx[ii] - int(nc / 2) + 1:idx[ii] + int(nc / 2), :], axis=0))
        s_cwt_z[ii, :] = np.squeeze(np.nanmean(s_cwt.z[idx[ii] - int(nc / 2) + 1:idx[ii] + int(nc / 2), :], axis=0))

    s_cwt_x = xr.DataArray(s_cwt_x, coords=[s_cwt_times, s_cwt.frequency], dims=["time", "frequency"])
    s_cwt_y = xr.DataArray(s_cwt_y, coords=[s_cwt_times, s_cwt.frequency], dims=["time", "frequency"])
    s_cwt_z = xr.DataArray(s_cwt_z, coords=[s_cwt_times, s_cwt.frequency], dims=["time", "frequency"])

    return s_cwt_x, s_cwt_y, s_cwt_z


def plot(axs, b_xyz, e_xyzfac_lf, e_xyzfac_hf, e_cwt_perp, e_cwt_para, b_cwt, pparam):
    plot_line(axs[0], b_xyz)
    axs[0].legend(["$B_x$", "$B_y$", "$B_z$"], ncol=3, frameon=False, loc="upper right")
    axs[0].set_ylabel("$B$ [nT]")

    plot_line(axs[1], e_xyzfac_lf)
    axs[1].legend(["$E_{\\perp 1}$", "$E_{\\perp 2}$", "$E_{\\parallel}$"], ncol=3, frameon=False, loc="upper right")
    axs[1].set_ylabel("$E$ [mV.m$^{-1}$]")
    axs[1].text(0.02, 0.83, "(b)", transform=axs[1].transAxes)

    plot_line(axs[2], e_xyzfac_hf)
    axs[2].legend(["$E_{\\perp 1}$", "$E_{\\perp 2}$", "$E_{\\parallel}$"], ncol=3, frameon=False, loc="upper right")
    axs[2].set_ylabel("$E$ [mV.m$^{-1}$]")

    axs[3], caxs3 = plot_spectr(axs[3], e_cwt_perp, cscale="log", yscale="log")
    plot_line(axs[3], pparam.f_lh)
    plot_line(axs[3], pparam.f_ce)
    plot_line(axs[3], pparam.f_pp)
    axs[3].set_ylabel("$f$ [Hz]")
    caxs3.set_ylabel("$E_{\\perp}^2$ " + "\n" + "[mV$^2$.m$^{-2}$.Hz$^{-1}$]")
    axs[3].legend(["$f_{lh}$", "$f_{ce}$", "$f_{pi}$"], ncol=3, loc="upper right", frameon=True)

    axs[4], caxs4 = plot_spectr(axs[4], e_cwt_para, cscale="log", yscale="log")
    plot_line(axs[4], pparam.f_lh)
    plot_line(axs[4], pparam.f_ce)
    plot_line(axs[4], pparam.f_pp)
    axs[4].set_ylabel("$f$ [Hz]")
    caxs4.set_ylabel("$E_{||}^2$ " + "\n" + "[mV$^2$.m$^{-2}$.Hz$^{-1}$]")
    axs[4].legend(["$f_{lh}$", "$f_{ce}$", "$f_{pi}$"], ncol=3, loc="upper right", frameon=True)

    axs[5], caxs5 = plot_spectr(axs[5], b_cwt, cscale="log", yscale="log")
    plot_line(axs[5], pparam.f_lh)
    plot_line(axs[5], pparam.f_ce)
    plot_line(axs[5], pparam.f_pp)
    axs[5].set_ylabel("$f$ [Hz]")
    caxs5.set_ylabel("$B^2$ " + "\n" + "[nT$^2$.Hz$^{-1}$]")
    axs[5].legend(["$f_{lh}$", "$f_{ce}$", "$f_{pi}$"], ncol=3, loc="upper right", frameon=True)

    axs[0].text(0.02, 0.83, "(a)", transform=axs[0].transAxes)
    axs[1].text(0.02, 0.83, "(b)", transform=axs[1].transAxes)
    axs[2].text(0.02, 0.83, "(c)", transform=axs[2].transAxes)
    axs[3].text(0.02, 0.83, "(d)", transform=axs[3].transAxes)
    axs[4].text(0.02, 0.83, "(e)", transform=axs[4].transAxes)
    axs[5].text(0.02, 0.83, "(f)", transform=axs[5].transAxes)

    fig = plt.gcf()
    fig.align_ylabels(axs)
    axs[-1].set_xlabel("2019-09-14 UTC")

    return axs


def main(tint, mms_id):
    # Load data
    # Background magnetic field from FGM
    b_xyz = get_data("B_gse_fgm_brst_l2", tint, mms_id)

    # Electric field from EDP
    e_xyz = get_data("E_gse_edp_brst_l2", tint, mms_id)

    # Magnetic field fluctuations from SCM
    b_scm = get_data("B_gse_scm_brst_l2", tint, mms_id)

    # Number density from FPI
    n_e = get_data("Ne_fpi_brst_l2", tint, mms_id)

    # Convert to field aligned coordinates
    e_xyzfac = convert_fac(e_xyz, b_xyz, [1, 0, 0])

    # Filter
    # Bandpass filter E and B waveforms
    fmin, fmax = [0.5, 1000]

    e_xyzfac_hf = filt(e_xyzfac, fmin, 0, 3)
    e_xyzfac_lf = filt(e_xyzfac, 0, fmin, 3)

    # Wavelet transforms
    # Wavelet transform field aligned electric field
    nf, nc = [100, 100]

    # Electric field
    e_cwt_x, e_cwt_y, e_cwt_z = ccwt(s=e_xyzfac, nf=nf, f=[fmin, fmax], nc=nc)

    e_cwt_perp = xr.DataArray(e_cwt_x + e_cwt_y, coords=[e_cwt_x.times.data, e_cwt_x.frequency.data],
                              dims=["time", "frequency"])

    e_cwt_para = xr.DataArray(e_cwt_z, coords=[e_cwt_x.times.data, e_cwt_x.frequency.data], dims=["time", "frequency"])

    # Magnetic field fluctuations
    b_cwt_x, b_cwt_y, b_cwt_z = ccwt(s=b_scm, nf=nf, f=[fmin, fmax], nc=nc)

    b_cwt = xr.DataArray(b_cwt_x + b_cwt_y + b_cwt_z, coords=[b_cwt_x.times.data, b_cwt_x.frequency.data],
                         dims=["time", "frequency"])

    # Compute plasma parameters
    pparam = plasma_calc(b_xyz, n_e, n_e, n_e, n_e)

    # Plot
    fig, axs = plt.subplots(6, sharex="all", figsize=(6.5, 9))
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.15, right=0.85, hspace=0.)

    axs = plot(axs, b_xyz, e_xyzfac_lf, e_xyzfac_hf, e_cwt_perp, e_cwt_para, b_cwt, pparam)

    axs[2].text(0.02, 0.15, "$f > ${:2.1f} Hz".format(fmin), transform=axs[2].transAxes)

    fig_name = "_".join([date_str(tint, 3), "ebfields.png"])
    fig.savefig(os.path.join("figures_examples", fig_name), format="png")

    plt.show()


if __name__ == "__main__":
    tint = ["2015-10-30T05:15:20.000", "2015-10-30T05:16:20.000"]

    mms_id = 3

    main(tint, mms_id)
