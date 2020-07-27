#!/usr/bin/env python
# coding: utf-8

# # Example_MMS_Polarizationanalysis
# Author : Louis RICHARD\
# E-Mail : louisr@irfu.se\
# Created : 2020-07-27

# In[1]:


from pyrfu import pyrf
from pyrfu import mms
from pyrfu import plot as pltrf

import numpy as np

from astropy import constants

import matplotlib.pyplot as plt


def load_data(Tint,ic):
	"""
	Load :
		Rxyz : spacecraft position
		Bxyz : Background magnetic field
		Exyz : Electric field
		Bscm : fluctuations of the magnetic field
	
	"""
	# Extend time interval for spacecraft position
	Tintl = pyrf.extend_tint(Tint,[-100,100])
	Rxyz = mms.get_data("R_gse",Tintl,ic)

	# load background magnetic field (FGM)
	Bxyz = mms.get_data("B_gse_fgm_brst_l2",Tint,ic)

	# load electric field (EDP)
	Exyz = mms.get_data("E_gse_edp_brst_l2",Tint,ic)

	# load fluctuations of the magnetic field (SCM)
	Bscm = mms.get_data("B_gse_scm_brst_l2",Tint,ic)

	return (Exyz,Bscm,Bxyz,Rxyz)

def polarization_analysis(Exyz,Bscm,Bxyz,Rxyz):
	"""
	Polarization analysis to compute :
		Bsum 		: the spectrogram of the total magnetic field power
		Esum 		: the spectrogram of the total electric field power
		ellipticity : the spectrogram of the ellipticity
		thetak 		: the spectrogram of the angle propagation angle
		dop 		: the spectrogram of the degree of polarization
		planarity 	: the spectrogram of the planarity
		vph 		: the spectrogram of the phase velocity
		pfluxz 		: the spectrogram of the Poynting flux

	"""

	# Compute electron cyclotron frequency
	Me 	= constants.m_e.value
	e 	= constants.e.value

	B_SI 		= pyrf.norm(Bxyz)*1e-9
	Wce 		= e*B_SI/Me
	ecfreq 		= Wce/(2*np.pi)

	# Polarization analysis
	polarization = pyrf.ebsp(Exyz,Bscm,Bxyz,Bxyz,Rxyz,[10,4000],fac=True,polarization=True)


	frequency 	= polarization["f"]
	time 		= polarization["t"]
	Bsum 		= polarization["bb_xxyyzzss"][...,3]
	Bperp 		= polarization["bb_xxyyzzss"][...,0]+polarization["bb_xxyyzzss"][...,1]
	Esum 		= polarization["ee_xxyyzzss"][...,3]
	Eperp	 	= polarization["ee_xxyyzzss"][...,0]+polarization["ee_xxyyzzss"][...,1]
	ellipticity = polarization["ellipticity"]
	dop 		= polarization["dop"]
	thetak 		= polarization["k_tp"][...,0]
	planarity 	= polarization["planarity"]
	pfluxz 		= polarization["pf_xyz"][...,2]/np.linalg.norm(polarization["pf_xyz"],axis=2)


	# Calculate phase speed v_ph = E/B.
	vph 	= np.sqrt(Esum/Bsum)*1e6
	vphperp = np.sqrt(Eperp/Bperp)*1e6


	# Remove points with very low B amplitudes
	Bsumthres = 1e-7
	removepts = Bsum.data < Bsumthres

	ellipticity.data[removepts] = np.nan
	thetak.data[removepts] 		= np.nan
	dop.data[removepts] 		= np.nan
	planarity.data[removepts]	= np.nan
	pfluxz.data[removepts] 		= np.nan
	vph.data[removepts] 		= np.nan
	vphperp.data[removepts] 	= np.nan

	return (Bsum,Esum,ellipticity,thetak,dop,planarity,vph,pfluxz,ecfreq)

def plot(Bsum,Esum,ellipticity,thetak,dop,planarity,vph,pfluxz,ecfreq):

	
	ecfreq01 	= ecfreq*0.1
	ecfreq05 	= ecfreq*0.5

	# Plot
	cmap = "jet"
	fig, axs = plt.subplots(8,sharex=True,figsize=(9,16))
	fig.subplots_adjust(bottom=0.1,top=0.95,left=0.15,right=0.85,hspace=0)

	# Magnetic field power spectrogram
	axs[0], caxs0 = pltrf.plot_spectr(axs[0],Bsum,yscale="log",cscale="log",cmap=cmap)
	pltrf.plot_line(axs[0],ecfreq,"w")
	pltrf.plot_line(axs[0],ecfreq01,"w")
	pltrf.plot_line(axs[0],ecfreq05,"w")
	axs[0].set_ylabel("$f$ [Hz]")
	caxs0.set_ylabel("$B^{2}$"+"\n"+"[nT$^2$.Hz$^{-1}$]")

	# Electric field power spectrogram
	axs[1], caxs1 = pltrf.plot_spectr(axs[1],Esum,yscale="log",cscale="log",cmap=cmap)
	pltrf.plot_line(axs[1],ecfreq,"w")
	pltrf.plot_line(axs[1],ecfreq01,"w")
	pltrf.plot_line(axs[1],ecfreq05,"w")
	axs[1].set_ylabel("$f$ [Hz]")
	caxs1.set_ylabel("$E^{2}$"+"\n"+"[mV$^2$.m$^{-2}$.Hz$^{-1}$]")

	# Ellipticity
	axs[2], caxs2 = pltrf.plot_spectr(axs[2],ellipticity,yscale="log",cscale="lin",cmap="seismic",clim=[-1,1])
	pltrf.plot_line(axs[2],ecfreq,"w")
	pltrf.plot_line(axs[2],ecfreq01,"w")
	pltrf.plot_line(axs[2],ecfreq05,"w")
	axs[2].set_ylabel("$f$ [Hz]")
	caxs2.set_ylabel("Ellipticity")

	# Theta k
	axs[3], caxs3 = pltrf.plot_spectr(axs[3],thetak*180/np.pi,yscale="log",cscale="lin",cmap=cmap,clim=[0,90])
	pltrf.plot_line(axs[3],ecfreq,"w")
	pltrf.plot_line(axs[3],ecfreq01,"w")
	pltrf.plot_line(axs[3],ecfreq05,"w")
	axs[3].set_ylabel("$f$ [Hz]")
	caxs3.set_ylabel("$\\theta_{k}$")

	# Degree of polariation
	axs[4], caxs4 = pltrf.plot_spectr(axs[4],dop,yscale="log",cscale="lin",cmap=cmap,clim=[0,1])
	pltrf.plot_line(axs[4],ecfreq,"w")
	pltrf.plot_line(axs[4],ecfreq01,"w")
	pltrf.plot_line(axs[4],ecfreq05,"w")
	axs[4].set_ylabel("$f$ [Hz]")
	caxs4.set_ylabel("DOP")

	# Planarity
	axs[5], caxs5 = pltrf.plot_spectr(axs[5],planarity,yscale="log",cscale="lin",cmap=cmap,clim=[0,1])
	pltrf.plot_line(axs[5],ecfreq,"w")
	pltrf.plot_line(axs[5],ecfreq01,"w")
	pltrf.plot_line(axs[5],ecfreq05,"w")
	axs[5].set_ylabel("$f$ [Hz]")
	caxs5.set_ylabel("planarity")

	# Phase velocity
	axs[6], caxs6 = pltrf.plot_spectr(axs[6],vph,yscale="log",cscale="log",cmap=cmap)
	pltrf.plot_line(axs[6],ecfreq,"w")
	pltrf.plot_line(axs[6],ecfreq01,"w")
	pltrf.plot_line(axs[6],ecfreq05,"w")
	axs[6].set_ylabel("$f$ [Hz]")
	caxs6.set_ylabel("$E/B$"+"\n"+"[m.$^{-1}$]")

	# Poynting flux
	axs[7], caxs7 = pltrf.plot_spectr(axs[7],pfluxz,yscale="log",cscale="lin",cmap="seismic",clim=[-1,1])
	pltrf.plot_line(axs[7],ecfreq,"w")
	pltrf.plot_line(axs[7],ecfreq01,"w")
	pltrf.plot_line(axs[7],ecfreq05,"w")
	axs[7].set_ylabel("$f$ [Hz]")
	caxs7.set_ylabel("$S_\\parallel/|S|$")


	plt.show()

if __name__ == "__main__":

	# Time interval
	Tint = ["2015-10-30T05:15:42.000","2015-10-30T05:15:54.000"]

	# Spacecraft index
	ic = 3

	Exyz, Bscm, Bxyz, Rxyz = load_data(Tint,ic)

	Bsum, Esum, ellipticity, thetak, dop, planarity, vph, pfluxz, ecfreq = polarization_analysis(Exyz,Bscm,Bxyz,Rxyz)

	plot(Bsum, Esum, ellipticity, thetak, dop, planarity, vph, pfluxz, ecfreq)

