# -*- coding: utf-8 -*-
"""
example_mms_b_e_j.py

Plots of B, J, E, JxB electric field, and J.E. Calculates J using curlometer method.

@author : Louis RICHARD
"""

import numpy as np
import matplotlib.pyplot as plt

from pyrfu.mms import get_data
from pyrfu.pyrf import avg_4sc, resample, c_4_j, convert_fac, dot
from pyrfu.plot import plot_line


def main(tint):
	# Load magnetic field
	b_mms = [get_data("B_dmpa_dfg_srvy_l2", tint, mms_id) for mms_id in range(1, 5)]
	b_xyz = avg_4sc(b_mms)

	# Load electric field
	e_mms = [get_data("E2d_dsl_edp_fast_l2", tint, mms_id) for mms_id in range(1, 5)]
	e_xyz = avg_4sc(e_mms)

	# Load H+ number density
	n_mms_i = [get_data("hplus_hpca_srvy_l2", tint, mms_id) for mms_id in range(1, 5)]
	n_i = avg_4sc(n_mms_i)
	n_i = resample(n_i, b_xyz)

	# Load spacecraft position
	r_mms = [get_data("R_gse", tint, mms_id) for mms_id in range(1, 5)]

	# Assuming GSE and DMPA are the same coordinate system.
	j_xyz, div_b, b_xyz_av, jxb_xyz, div_t_shear, div_pb = c_4_j(r_mms, b_mms)

	div_curl = div_b
	div_curl.data = np.abs(div_b.data) / np.linalg.norm(j_xyz)

	jxb_xyz.data /= n_i.data[:, np.newaxis]
	jxb_xyz.data /= 1.6e-19 * 1000  			# Convert to (mV/m)

	jxb_xyz.data[np.linalg.norm(jxb_xyz.data, axis=1) > 100] = np.nan  # Remove some questionable fields

	# Transform current density into field-aligned coordinates
	j_xyzfac = convert_fac(j_xyz, b_xyz_av, [1, 0, 0])

	# Compute dissipation terms
	j_xyz = resample(j_xyz, e_xyz)
	e_dot_j = dot(e_xyz, j_xyz) / 1000  # J (nA/m^2), E (mV/m), E.J (nW/m^3)

	# plot
	fig, axs = plt.subplots(8, sharex="all", figsize=(6.5, 9))
	fig.subplots_adjust(bottom=0.1, top=0.95, left=0.15, right=0.85, hspace=0)

	plot_line(axs[0], b_xyz_av)
	axs[0].set_ylabel("$B_{DMPA}$" + "\n" + "[nT]")
	axs[0].legend(["$B_x$", "$B_y$", "$B_z$"], frameon=False, ncol=3, loc="upper right")
	axs[0].set_ylim([-70, 70])
	axs[0].text(0.02, 0.83, "(a)", transform=axs[0].transAxes)

	for n in n_mms_i:
		plot_line(axs[1], n)

	axs[1].set_ylabel("$n_i$" + "\n" + "[cm$^{-3}$]")
	axs[1].set_yscale("log")
	axs[1].set_ylim([1e-4, 1e1])
	axs[1].text(0.02, 0.83, "(b)", transform=axs[1].transAxes)
	axs[1].legend(["MMS1", "MMS2", "MMS3", "MMS4"], frameon=False, ncol=4, loc="upper right")

	j_xyz.data *= 1e9
	plot_line(axs[2], j_xyz)
	axs[2].set_ylabel("$J_{DMPA}$" + "\n" + "[nA.m$^{-2}$]")
	axs[2].legend(["$J_x$", "$J_y$", "$J_z$"], frameon=False, ncol=3, loc="upper right")
	axs[2].text(0.02, 0.83, "(c)", transform=axs[2].transAxes)

	j_xyzfac.data *= 1e9
	plot_line(axs[3], j_xyzfac)
	axs[3].set_ylabel("$J_{FAC}$" + "\n" + "[nA.m$^{-2}$]")
	axs[3].legend(["$J_{\\perp 1}$", "$J_{\\perp 2}$", "$J_{\\parallel}$"], frameon=False, ncol=3, loc="upper right")
	axs[3].text(0.02, 0.83, "(d)", transform=axs[3].transAxes)

	plot_line(axs[4], div_curl)
	axs[4].set_ylabel("$\\frac{|\\nabla . B|}{|\\nabla \\times B|}$")
	axs[4].text(0.02, 0.83, "(e)", transform=axs[4].transAxes)

	plot_line(axs[5], e_xyz)
	axs[5].set_ylabel("$E_{DSL}$" + "\n" + "[mV.m$^{-1}$]")
	axs[5].legend(["$E_x$", "$E_y$", "$E_z$"], frameon=False, ncol=3, loc="upper right")
	axs[5].text(0.02, 0.83, "(f)", transform=axs[5].transAxes)

	plot_line(axs[6], jxb_xyz)
	axs[6].set_ylabel("$J \\times B/n_{e}q_{e}$" + "\n" + "[mV.m$^{-1}$]")
	axs[6].text(0.02, 0.83, "(g)", transform=axs[6].transAxes)

	plot_line(axs[7], e_dot_j)
	axs[7].set_ylabel("$E . J$" + "\n" + "[nW.m$^{-3}]$")
	axs[7].text(0.02, 0.83, "(h)", transform=axs[7].transAxes)

	axs[0].set_title("MMS - Current density and fields")
	fig.align_ylabels(axs)
	axs[-1].set_xlim(tint)


if __name__ == "__main__":
	tint = ["2016-06-07T01:53:44.000", "2016-06-07T19:30:34.000"]

	main(tint)
