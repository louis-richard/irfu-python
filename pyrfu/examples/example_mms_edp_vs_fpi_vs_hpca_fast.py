# -*- coding: utf-8 -*-
"""
example_mms_edp_vs_fpi_vs_hpca_fast.py

@author : Louis RICHARD
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from pyrfu import mms, pyrf
from pyrfu.plot import plot_line


def main(tint, mms_id):
    # Load data
    # FGM/DFG
    b_dmpa_fgm_srvy_l2 = mms.get_data("B_dmpa_fgm_srvy_l2", tint, mms_id)

    e_dsl_edp_l2 = mms.get_data("E_dsl_edp_fast_l2", tint, mms_id)

    e2d_dsl_edp_l2pre = mms.get_data("E2d_dsl_edp_fast_l2pre", tint, mms_id)

    e_adp_edp = mms.get_data("E_ssc_edp_fast_l1b", tint, mms_id)
    e_adp_edp = -e_adp_edp[:, 2] * 1.5

    # FPI
    v_i_dbcs_fpi = mms.get_data("Vi_dbcs_fpi_fast_l2", tint, mms_id)
    v_e_dbcs_fpi = mms.get_data("Ve_dbcs_fpi_fast_l2", tint, mms_id)
    n_e_fpi = mms.get_data("Ne_fpi_fast_l2", tint, mms_id)

    v_e_dbcs_fpi.data[n_e_fpi.data < 0.06, :] = np.nan

    # HPCA
    v_hplus_dbcs_hpca = mms.get_data("Vhplus_dbcs_hpca_srvy_l2", tint, mms_id)

    # correct Ez in E2d
    # XXX: this should be undone later
    e2d_dsl_edp_l2pre, d = pyrf.edb(e2d_dsl_edp_l2pre, b_dmpa_fgm_srvy_l2, 10, "Eperp+NaN")

    # Comp VxB
    [v_para_i, v_perp_i, alpha] = pyrf.dec_parperp(v_i_dbcs_fpi, b_dmpa_fgm_srvy_l2)
    [v_para_e, v_perp_e, alpha] = pyrf.dec_parperp(v_e_dbcs_fpi, b_dmpa_fgm_srvy_l2)

    # ExB drift
    vexb_xyz = pyrf.e_vxb(e_dsl_edp_l2, b_dmpa_fgm_srvy_l2, -1)
    vexb_xyz_l2pre = pyrf.e_vxb(e2d_dsl_edp_l2pre, b_dmpa_fgm_srvy_l2, -1)

    # Convection electric field
    evxb_xyz_i = pyrf.e_vxb(v_i_dbcs_fpi, pyrf.resample(b_dmpa_fgm_srvy_l2, v_i_dbcs_fpi))
    evxb_xyz_e = pyrf.e_vxb(v_e_dbcs_fpi, pyrf.resample(b_dmpa_fgm_srvy_l2, v_e_dbcs_fpi))

    [v_para_hplus, v_perp_hplus, alpha] = pyrf.dec_parperp(v_hplus_dbcs_hpca, b_dmpa_fgm_srvy_l2)
    evxb_xyz_hplus = pyrf.e_vxb(v_hplus_dbcs_hpca, pyrf.resample(b_dmpa_fgm_srvy_l2, v_hplus_dbcs_hpca))

    # plots
    # Plot electric field
    fig, axs = plt.subplots(4, sharex="all", figsize=(16, 9))
    fig.subplots_adjust(bottom=.1, top=.9, left=.1, right=.9, hspace=0.)

    plot_line(axs[0], b_dmpa_fgm_srvy_l2)
    axs[0].set_title("MMS{:d}".format(mms_id))
    axs[0].set_ylabel("$B$ DSL [nT]")
    axs[0].legend(["X", "Y", "Z"], ncol=3, frameon=False, loc="upper right")

    for i in range(3):
        plot_line(axs[i + 1], e2d_dsl_edp_l2pre[:, i])
        plot_line(axs[i + 1], e_dsl_edp_l2[:, i])
        plot_line(axs[i + 1], evxb_xyz_e[:, i])
        plot_line(axs[i + 1], evxb_xyz_i[:, i])
        plot_line(axs[i + 1], evxb_xyz_hplus[:, i])
        axs[i + 1].legend(["E L2pre", "E l2", "$V_{e}\\times B$", "$V_{i}\\times B$", "$V_{H+}\\times B$"], ncol=5,
                          frameon=False, loc="upper right")

    plot_line(axs[-1], e_adp_edp)
    axs[-1].legend(["E L2pre", "E l2", "$V_{e}\\times B$", "$V_{i}\\times B$", "$V_{H+}\\times B$", "E ADP"], ncol=6,
                   frameon=False, loc="upper right")

    axs[1].set_ylabel("$E_x$ DSL [mV.m$^{-1}$]")
    axs[2].set_ylabel("$E_x$ DSL [mV.m$^{-1}$]")
    axs[3].set_ylabel("$E_x$ DSL [mV.m$^{-1}$]")

    fig.align_ylabels(axs)
    axs[-1].set_xlabel("{} UTC".format(tint[0][:10]))
    axs[-1].set_xlim(tint)

    fig_name = "_".join([pyrf.fname(tint, 3), "e_edp_fast_vs_fpi_fast_vs_hpca_fast.png"])
    fig.savefig(os.path.join("figures_examples", fig_name), format="png")

    # Plot velocity
    fig, axs = plt.subplots(4, sharex="all", figsize=(16, 9))
    fig.subplots_adjust(bottom=.1, top=.9, left=.1, right=.9, hspace=0.)

    plot_line(axs[0], b_dmpa_fgm_srvy_l2)
    axs[0].set_title("MMS{:d}".format(mms_id))
    axs[0].set_ylabel("$B$ DSL [nT]")
    axs[0].legend(["X", "Y", "Z"], ncol=3, frameon=False, loc="upper right")

    for i in range(3):
        plot_line(axs[i + 1], vexb_xyz_l2pre[:, i])
        plot_line(axs[i + 1], vexb_xyz[:, i])
        plot_line(axs[i + 1], v_perp_e[:, i])
        plot_line(axs[i + 1], v_perp_i[:, i])
        plot_line(axs[i + 1], v_perp_hplus[:, i])
        axs[i + 1].legend(["VExE L2pre", "VExE", "$V_{e,\\perp}$", "$V_{i,\\perp}$", "$V_{H+,\\perp}$"], ncol=5,
                          frameon=False, loc="upper right")

    axs[1].set_ylabel("$V_x$ DSL [km.s$^{-1}$]")
    axs[2].set_ylabel("$V_y$ DSL [km.s$^{-1}$]")
    axs[3].set_ylabel("$V_z$ DSL [km.s$^{-1}$]")

    fig.align_ylabels(axs)
    axs[-1].set_xlabel("{} UTC".format(tint[0][:10]))
    axs[-1].set_xlim(tint)

    fig_name = "_".join([pyrf.fname(tint, 3), "vexb_edp_fast_vs_fpi_fast_vs_hpca_fast.png"])
    fig.savefig(os.path.join("figures_examples", fig_name), format="png")

    plt.show()


if __name__ == "__main__":
    # Define time interval
    tint = ["2016-08-10T09:50:00", "2016-08-10T10:15:00"]

    # Spacecraft index
    mms_id = 1
