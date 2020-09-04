#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
example_mms_quickplot_all_sum.py

@author : Louis RICHARD
"""

import matplotlib.pyplot as plt

from pyrfu import mms
from pyrfu.plot import plot_line, plot_spectr


def main(tint, mms_id):
    # Magnetic field
    b_xyz = mms.get_data("B_gse_fgm_srvy_l2", tint, mms_id)

    # Electric field
    e_xyz = mms.get_data("E_gse_edp_fast_l2", tint, mms_id)

    # Density energy flux
    enflux_omni_i, enflux_omni_e = [mms.get_data("Enflux{}_fpi_fast_l2".format(s), tint, mms_id) for s in ["i", "e"]]

    # Number density
    n_i, n_e = [mms.get_data("N{}_fpi_fast_l2".format(s), tint, mms_id) for s in ["i", "e"]]

    # Ion bulk velocity
    v_xyz_i, v_xyz_e = [mms.get_data("V{}_gse_fpi_fast_l2".format(s), tint, mms_id) for s in ["i", "e"]]

    fig, axs = plt.subplots(6, sharex="all", figsize=(6.5, 9))
    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.15, right=0.85, hspace=0)

    axs[0] = plot_line(axs[0], b_xyz)

    axs[1], caxs1 = plot_spectr(axs[1], enflux_omni_i, yscale="log", cscale="log", cmap="viridis")

    axs[2], caxs1 = plot_spectr(axs[2], enflux_omni_e, yscale="log", cscale="log", cmap="viridis")

    plot_line(axs[3], n_i)
    plot_line(axs[3], n_e)

    plot_line(axs[4], v_xyz_i)
    plot_line(axs[4], v_xyz_e)

    plot_line(axs[5], e_xyz)

    plt.show()


if __name__ == "__main__":
    # Define time interval
    tint = ["2019-09-14T06:00:00.000", "2019-09-14T08:00:00.000"]

    # Spacecraft index
    mms_id = 1

    main(tint, mms_id)
