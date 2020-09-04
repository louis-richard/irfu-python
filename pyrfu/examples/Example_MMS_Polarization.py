#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
example_mms_polarization.py

@author : Louis RICHARD
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from pyrfu import pyrf
from pyrfu import mms
from pyrfu.plot import plot_line, plot_spectr
from astropy import constants


def main(tint, mms_id):
	# Extend time interval for spacecraft position
	tint_long = pyrf.extend_tint(tint, [-100, 100])
	r_xyz = mms.get_data("R_gse", tint_long, mms_id)

	# load background magnetic field (FGM)
	b_xyz = mms.get_data("B_gse_fgm_brst_l2", tint, mms_id)

	# load electric field (EDP)
	e_xyz = mms.get_data("E_gse_edp_brst_l2", tint, mms_id)

	# load fluctuations of the magnetic field (SCM)
	b_scm = mms.get_data("B_gse_scm_brst_l2", tint, mms_id)

	# Compute electron cyclotron frequency
	me = constants.m_e.value
	e = constants.e.value

	b_si = pyrf.norm(b_xyz) * 1e-9
	w_ce = e * b_si / me
	f_ce = w_ce / (2 * np.pi)

	# Polarization analysis
	polarization = pyrf.ebsp(e_xyz, b_scm, b_xyz, b_xyz, r_xyz, [10, 4000], fac=True, polarization=True)

	# Unpack data
	# Spectrogram of total magnetic and electric fields
	b_sum = polarization["bb_xxyyzzss"][..., 3]
	e_sum = polarization["ee_xxyyzzss"][..., 3]

	# Spectrogram of perpandicular magnetic and electric fields
	b_perp = polarization["bb_xxyyzzss"][..., 0] + polarization["bb_xxyyzzss"][..., 1]
	e_perp = polarization["ee_xxyyzzss"][..., 0] + polarization["ee_xxyyzzss"][..., 1]

	# Ellipticity
	ellipticity = polarization["ellipticity"]

	# Degree of polarization
	dop = polarization["dop"]

	# Polarization angle
	theta_k = polarization["k_tp"][..., 0]

	# Planarity
	planarity = polarization["planarity"]

	# Poynting flux
	pflux_z = polarization["pf_xyz"][..., 2] / np.linalg.norm(polarization["pf_xyz"], axis=2)

	# Calculate phase speed v_ph = E/B.
	vph = np.sqrt(e_sum / b_sum) * 1e6
	vph_perp = np.sqrt(e_perp / b_perp) * 1e6

	# Remove points with very low B amplitudes
	b_sum_thres = 1e-7
	remove_pts = b_sum.data < b_sum_thres

	ellipticity.data[remove_pts] = np.nan
	theta_k.data[remove_pts] = np.nan
	dop.data[remove_pts] = np.nan
	planarity.data[remove_pts] = np.nan
	pflux_z.data[remove_pts] = np.nan
	vph.data[remove_pts] = np.nan
	vph_perp.data[remove_pts] = np.nan

	# Plot
	f_ce_01, f_ce_05 = [f_ce * 0.1, f_ce * 0.5]

	# Plot
	cmap = "jet"
	fig, axs = plt.subplots(8, sharex=" all", figsize=(9, 16))
	fig.subplots_adjust(bottom=0.1, top=0.95, left=0.15, right=0.85, hspace=0)

	# Magnetic field power spectrogram
	axs[0], caxs0 = plot_spectr(axs[0], b_sum, yscale="log", cscale="log", cmap=cmap)
	plot_line(axs[0], f_ce, "w")
	plot_line(axs[0], f_ce_01, "w")
	plot_line(axs[0], f_ce_05, "w")
	axs[0].set_ylabel("$f$ [Hz]")
	caxs0.set_ylabel("$B^{2}$" + "\n" + "[nT$^2$.Hz$^{-1}$]")

	# Electric field power spectrogram
	axs[1], caxs1 = plot_spectr(axs[1], e_sum, yscale="log", cscale="log", cmap=cmap)
	plot_line(axs[1], f_ce, "w")
	plot_line(axs[1], f_ce_01, "w")
	plot_line(axs[1], f_ce_05, "w")
	axs[1].set_ylabel("$f$ [Hz]")
	caxs1.set_ylabel("$E^{2}$" + "\n" + "[mV$^2$.m$^{-2}$.Hz$^{-1}$]")

	# Ellipticity
	axs[2], caxs2 = plot_spectr(axs[2], ellipticity, yscale="log", cscale="lin", cmap="seismic", clim=[-1, 1])
	plot_line(axs[2], f_ce, "w")
	plot_line(axs[2], f_ce_01, "w")
	plot_line(axs[2], f_ce_05, "w")
	axs[2].set_ylabel("$f$ [Hz]")
	caxs2.set_ylabel("Ellipticity")

	# Theta k
	axs[3], caxs3 = plot_spectr(axs[3], theta_k * 180 / np.pi, yscale="log", cscale="lin", cmap=cmap,
									  clim=[0, 90])
	plot_line(axs[3], f_ce, "w")
	plot_line(axs[3], f_ce_01, "w")
	plot_line(axs[3], f_ce_05, "w")
	axs[3].set_ylabel("$f$ [Hz]")
	caxs3.set_ylabel("$\\theta_{k}$")

	# Degree of polariation
	axs[4], caxs4 = plot_spectr(axs[4], dop, yscale="log", cscale="lin", cmap=cmap, clim=[0, 1])
	plot_line(axs[4], f_ce, "w")
	plot_line(axs[4], f_ce_01, "w")
	plot_line(axs[4], f_ce_05, "w")
	axs[4].set_ylabel("$f$ [Hz]")
	caxs4.set_ylabel("DOP")

	# Planarity
	axs[5], caxs5 = plot_spectr(axs[5], planarity, yscale="log", cscale="lin", cmap=cmap, clim=[0, 1])
	plot_line(axs[5], f_ce, "w")
	plot_line(axs[5], f_ce_01, "w")
	plot_line(axs[5], f_ce_05, "w")
	axs[5].set_ylabel("$f$ [Hz]")
	caxs5.set_ylabel("planarity")

	# Phase velocity
	axs[6], caxs6 = plot_spectr(axs[6], vph, yscale="log", cscale="log", cmap=cmap)
	plot_line(axs[6], f_ce, "w")
	plot_line(axs[6], f_ce_01, "w")
	plot_line(axs[6], f_ce_05, "w")
	axs[6].set_ylabel("$f$ [Hz]")
	caxs6.set_ylabel("$E/B$" + "\n" + "[m.$^{-1}$]")

	# Poynting flux
	axs[7], caxs7 = plot_spectr(axs[7], pflux_z, yscale="log", cscale="lin", cmap="seismic", clim=[-1, 1])
	plot_line(axs[7], f_ce, "w")
	plot_line(axs[7], f_ce_01, "w")
	plot_line(axs[7], f_ce_05, "w")
	axs[7].set_ylabel("$f$ [Hz]")
	caxs7.set_ylabel("$S_\\parallel/|S|$")

	fig_name = "_".join([pyrf.fname(tint, 3), "ebfields.png"])
	fig.savefig(os.path.join("figures_examples", fig_name), format="png")

	plt.show()


if __name__ == "__main__":
	# Time interval
	tint = ["2015-10-30T05:15:42.000", "2015-10-30T05:15:54.000"]

	# Spacecraft index
	mms_id = 3

	main(tint, mms_id)
