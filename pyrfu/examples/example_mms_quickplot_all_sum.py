#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
example_mms_quickplot_all_sum.py

@author : Louis RICHARD
"""

import argparse
import matplotlib.pyplot as plt

from pyrfu import mms
from pyrfu.plot import plot_line, plot_spectr


def main(args):
    # Magnetic field
    b_xyz = mms.get_data("B_gse_fgm_srvy_l2", args.tint, args.mms_id, args.verbose)

    # Electric field
    e_xyz = mms.get_data("E_gse_edp_fast_l2", args.tint, args.mms_id, args.verbose)

    # Density energy flux
    def_omni_i, def_omni_e = [mms.get_data("DEF{}_fpi_fast_l2".format(s), args.tint, args.mms_id, args.verbose) for s
                              in ["i", "e"]]

    # Number density
    n_i, n_e = [mms.get_data("N{}_fpi_fast_l2".format(s), args.tint, args.mms_id, args.verbose) for s in ["i", "e"]]

    # Ion bulk velocity
    v_xyz_i, v_xyz_e = [mms.get_data("V{}_gse_fpi_fast_l2".format(s), args.tint, args.mms_id, args.verbose) for s in
                        ["i", "e"]]

    fig, axs = plt.subplots(6, sharex="all", figsize=(6.5, 9))
    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.15, right=0.85, hspace=0)

    axs[0] = plot_line(axs[0], b_xyz)

    axs[1], caxs1 = plot_spectr(axs[1], def_omni_i, yscale="log", cscale="log", cmap="viridis")

    axs[2], caxs1 = plot_spectr(axs[2], def_omni_e, yscale="log", cscale="log", cmap="viridis")

    plot_line(axs[3], n_i)
    plot_line(axs[3], n_e)

    plot_line(axs[4], v_xyz_i)
    plot_line(axs[4], v_xyz_e)

    plot_line(axs[5], e_xyz)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data rate
    parser.add_argument("-b",
                        "--brst",
                        help="Use burst data default is srvy",
                        action="store_true")
    # Spacecraft index
    parser.add_argument("-m",
                        "--mms_id",
                        help="Spacecraft index default is 1",
                        type=int,
                        default=1,
                        action="store")
    # Verbosity
    parser.add_argument("-v",
                        "--verbose",
                        help="Increase verbosity",
                        action="store_true")

    main(parser.parse_args())
