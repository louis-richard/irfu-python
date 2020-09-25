# -*- coding: utf-8 -*-
"""
example_mms_edp_vs_fpi_vs_hpca_brst.py

@author : Louis RICHARD
"""

import os
import matplotlib.pyplot as plt

from pyrfu import mms, pyrf
from pyrfu.plot import plot_line


def main(tint, mms_id):
    # Load data
    # FGM/DFG
    b_dmpa_fgm_srvy = mms.get_data("B_dmpa_fgm_srvy_l2", tint, mms_id)

    if b_dmpa_fgm_srvy is None:
        print("loading l2pre DFG\n")
        b_dmpa_fgm_srvy = mms.get_data("B_dmpa_dfg_srvy_l2pre", tint, mms_id)

        if b_dmpa_fgm_srvy is None:
            print("loading QL DFG\n")
            b_dmpa_fgm_srvy = mms.get_data("B_dmpa_dfg_srvy_ql", tint, mms_id)

    # EDP
    e_dsl_edp = mms.get_data("E_dsl_edp_brst_l2", tint, mms_id)
    if e_dsl_edp is None:
        print("loading QL DCE\n")
        e_dsl_edp = mms.get_data("E_dsl_edp_brst_ql", tint, mms_id)

    # In spin plane electric field
    e2d_dsl_edp = mms.get_data("E2d_dsl_edp_brst_l2pre", tint, mms_id)
    if e2d_dsl_edp is None:
        print("loading QL DCE2d\n")
        e2d_dsl_edp = mms.get_data("E2d_dsl_edp_brst_ql", tint, mms_id)

    # ADP
    e_adp_edp = mms.get_data("E_ssc_edp_brst_l1b", tint, mms_id)
    e_adp_edp = -e_adp_edp[:, 2] * 1.5

    # FPI
    v_i_dbcs_fpi, v_e_dbcs_fpi = [mms.get_data("V{}_dbcs_fpi_brst_l2".format(s), tint, mms_id) for s in ["i", "e"]]

    # HPCA
    v_hplus_dbcs_hpca = mms.get_data("Vhplus_dbcs_hpca_brst_l2", tint, mms_id)
    if v_hplus_dbcs_hpca is None:
        v_hplus_dbcs_hpca = mms.get_data("Vhplus_dbcs_hpca_brst_l1b", tint, mms_id)

    # Decompose parallel and perpandicular components
    v_para_i, v_perp_i, _ = pyrf.dec_par_perp(v_i_dbcs_fpi, b_dmpa_fgm_srvy)
    v_para_e, v_perp_e, _ = pyrf.dec_par_perp(v_e_dbcs_fpi, b_dmpa_fgm_srvy)

    e_para, e_perp, _ = pyrf.dec_par_perp(e_dsl_edp, b_dmpa_fgm_srvy)

    v_para_hplus, v_perp_hplus, _ = pyrf.dec_par_perp(v_hplus_dbcs_hpca, b_dmpa_fgm_srvy)

    # Compute velocity from electric fields
    vexb_xyz = pyrf.e_vxb(e_dsl_edp, b_dmpa_fgm_srvy, -1)
    ve2dxb_xyz = pyrf.e_vxb(e2d_dsl_edp, b_dmpa_fgm_srvy, -1)

    # Compute electric field from velocities
    evxb_xyz_i = pyrf.e_vxb(v_i_dbcs_fpi, pyrf.resample(b_dmpa_fgm_srvy, v_i_dbcs_fpi))
    evxb_xyz_e = pyrf.e_vxb(v_e_dbcs_fpi, pyrf.resample(b_dmpa_fgm_srvy, v_e_dbcs_fpi))
    evxb_xyz_hplus = pyrf.e_vxb(v_hplus_dbcs_hpca, pyrf.resample(b_dmpa_fgm_srvy, v_hplus_dbcs_hpca))

    # plot
    fig, axs = plt.subplots(3, sharex="all", figsize=(16, 9))
    fig.subplots_adjust(bottom=.1, top=.9, left=.1, right=.9, hspace=0.05)

    for i in range(3):
        plot_line(axs[i], vexb_xyz[:, i])
        plot_line(axs[i], ve2dxb_xyz[:, i])
        plot_line(axs[i], v_perp_e[:, i])
        plot_line(axs[i], v_perp_i[:, i])
        plot_line(axs[i], v_perp_hplus[:, i])
        axs[i].legend(["VExB", "VE2dxB", "$V_{e,\\perp}$", "$V_{i,\\perp}$", "$V_{H+,\\perp}$"], ncol=5, frameon=False,
                      loc="upper right")

    axs[0].set_ylabel("$V_x$ DSL [km.s$^{-1}$]")
    axs[1].set_ylabel("$V_y$ DSL [km.s$^{-1}$]")
    axs[2].set_ylabel("$V_z$ DSL [km.s$^{-1}$]")

    axs[-1].set_xlim(tint)
    axs[0].set_title("MMS{:d}".format(mms_id))
    axs[-1].set_xlabel("{} UTC".format(tint[0][:10]))

    fig_name = "_".join([pyrf.date_str(tint, 3), "vexb_edp_vs_fpi_vs_hpca_brst.png"])
    fig.savefig(os.path.join("figures_examples", fig_name), format="png")

    fig, axs = plt.subplots(3, sharex="all", figsize=(16, 9))
    fig.subplots_adjust(bottom=.1, top=.9, left=.1, right=.9, hspace=0.05)

    for i in range(3):
        plot_line(axs[i], e2d_dsl_edp[:, i])
        plot_line(axs[i], e_dsl_edp[:, i])
        plot_line(axs[i], e_perp[:, i])
        plot_line(axs[i], evxb_xyz_e[:, i])
        plot_line(axs[i], evxb_xyz_i[:, i])
        plot_line(axs[i], evxb_xyz_hplus[:, i])

    plot_line(axs[2], e_adp_edp)

    axs[0].legend(["$E2d$", "$E$", "$E_\\perp$", "$V_{e}\\times B$", "$V_{i}\\times B$", "$V_{H+}\\times B$"], ncol=6,
                  frameon=False, loc="upper right")
    axs[1].legend(["$E2d$", "$E$", "$E_\\perp$", "$V_{e}\\times B$", "$V_{i}\\times B$", "$V_{H+}\\times B$"], ncol=6,
                  frameon=False, loc="upper right")
    axs[2].legend(
        ["$E2d$", "$E$", "$E_\\perp$", "$V_{e}\\times B$", "$V_{i}\\times B$", "$V_{H+}\\times B$", "$E$ adp"], ncol=7,
        frameon=False, loc="upper right")

    axs[0].set_ylabel("$E_x$ DSL [mV.m$^{-1}$]")
    axs[1].set_ylabel("$E_y$ DSL [mV.m$^{-1}$]")
    axs[2].set_ylabel("$E_z$ DSL [mV.m$^{-1}$]")

    axs[-1].set_xlim(tint)
    axs[0].set_title("MMS{:d}".format(mms_id))
    axs[-1].set_xlabel("{} UTC".format(tint[0][:10]))

    fig_name = "_".join([pyrf.date_str(tint, 3), "e_edp_vs_fpi_vs_hpca_brst.png"])
    fig.savefig(os.path.join("figures_examples", fig_name), format="png")

    plt.show()


if __name__ == "__main__":
    # Define time interval
    tint = ["2019-09-14T07:54:00", "2019-09-14T08:11:00"]

    # Spacecraft index
    mms_id = 4

    main(tint, mms_id)
