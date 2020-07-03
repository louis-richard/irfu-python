# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Title 	: Example_MMS_Polarization.py
# Author 	: L. RICHARD
# E-Mail 	: louisr@irfu.se
# Created 	: 03-June-2020
# Updated 	: 04-June-2020
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Notes : 
#
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


from pyrfu import pyrf
from pyrfu import mms
from pyrfu import plot as pltrf
import numpy as np
import pdb
from astropy import constants
import seaborn as sns
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
date_form = mdates.DateFormatter("%H:%M:%S")
sns.set()
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8,'axes.grid': True,\
                        'font.family': ['sans-serif']})


mode = "paper"
if mode == "paper":
    sns.set_context(mode)
    fig_size= (6.5,9)
elif mode == "talk":
    sns.set_context("paper")
    fig_size= (16,9)
plt.rc('lines', linewidth=0.8)
color = ["k","b","r","g","c","m","y"]
default_cycler = cycler(color=color)
plt.rc('axes',prop_cycle=default_cycler)

plt.close("all")

# path of MMS data
data_path = "/Volumes/mms"

ic = 3 # Spacecraft number

# Define time interval
Tint = ["2015-10-30T05:15:42.000","2015-10-30T05:15:54.000"]

# Extent time interval for spacecraft position
Tintl = pyrf.extend_tint(Tint,[-100,100])

# Spacecraft position
exec("Rxyz = pyrf.get_data('R_gse',Tintl,?)".replace("?",str(ic)))
# Background magnetic field
exec("Bxyz = pyrf.get_data('B_gse_fgm_brst_l2',Tint,?)".replace("?",str(ic)))
# Electric field
exec("Exyz = pyrf.get_data('E_gse_edp_brst_l2',Tint,?)".replace("?",str(ic)))
# Magnetic field fluctuations
exec("Bscm = pyrf.get_data('B_gse_scm_brst_l2',Tint,?)".replace("?",str(ic)))

# Polarization analysis
Me 			= constants.m_e.value
e 			= constants.e.value
B_SI 		= pyrf.abs(Bxyz)*1e-9
Wce 		= e*B_SI/Me
ecfreq 		= Wce/(2*np.pi)
ecfreq01 	= ecfreq*.1
ecfreq05 	= ecfreq*.5



polarization = pyrf.ebsp(Exyz,Bscm,Bxyz,Bxyz,Rxyz,[10,4000],polarization=True,fac=True)
#polarization = pyrf.ebsp(Exyz,Bscm,Bxyz,Bxyz,Rxyz,[10,4000],fac=True)


Bsum 	= polarization["bb_xxyyzzss"][...,3]
Bperp 	= polarization["bb_xxyyzzss"][...,0]+polarization["bb_xxyyzzss"][...,1]
Esum 	= polarization["ee_xxyyzzss"][...,3]
Eperp 	= polarization["ee_xxyyzzss"][...,0]+polarization["ee_xxyyzzss"][...,1]


ellipticity = polarization["ellipticity"]
dop 		= polarization["dop"]
thetak 		= polarization["k_tp"][...,0]
planarity 	= polarization["planarity"]
pfluxz 		= polarization["pf_xyz"][...,2]/np.sqrt(polarization["pf_xyz"][...,0]**2+polarization["pf_xyz"][...,1]**2+polarization["pf_xyz"][...,2]**2);




# Calculate phase speed v_ph = E/B.
vph 	= np.sqrt(Esum/Bsum)*1e6
vphperp = np.sqrt(Eperp/Bperp)*1e6


# Remove points with very low B amplitutes
Bsumthreshold 								= 1e-7
ellipticity.data[Bsum.data<Bsumthreshold] 	= np.nan
thetak.data[Bsum.data<Bsumthreshold]		= np.nan
dop.data[Bsum.data<Bsumthreshold]			= np.nan
planarity.data[Bsum.data<Bsumthreshold]		= np.nan
pfluxz.data[Bsum.data<Bsumthreshold] 		= np.nan
vph.data[Bsum.data<Bsumthreshold] 			= np.nan
vphperp.data[Bsum.data<Bsumthreshold] 		= np.nan


fig, axs = plt.subplots(8,sharex=True,figsize=(6.5,9))
fig.subplots_adjust(bottom=0.1,top=0.95,left=0.15,right=0.85,hspace=0)
axs[0], caxs0 = pyrf.plot_spectr(axs[0],Bsum,yscale="log",cscale="log")
pyrf.plot(axs[0],ecfreq,"w")
pyrf.plot(axs[0],ecfreq01,"w")
pyrf.plot(axs[0],ecfreq05,"w")
axs[0].set_ylabel("f [Hz]")

axs[1], caxs1 = pyrf.plot_spectr(axs[1],Esum,yscale="log",cscale="log")
pyrf.plot(axs[1],ecfreq,"w")
pyrf.plot(axs[1],ecfreq01,"w")
pyrf.plot(axs[1],ecfreq05,"w")
axs[1].set_ylabel("f [Hz]")


axs[2], caxs2 = pyrf.plot_spectr(axs[2],ellipticity,yscale="log")
pyrf.plot(axs[2],ecfreq,"w")
pyrf.plot(axs[2],ecfreq01,"w")
pyrf.plot(axs[2],ecfreq05,"w")
axs[2].set_ylabel("f [Hz]")
caxs2.set_ylabel("ellipticity")

axs[3], caxs3 = pyrf.plot_spectr(axs[3],thetak,yscale="log")
pyrf.plot(axs[3],ecfreq,"w")
pyrf.plot(axs[3],ecfreq01,"w")
pyrf.plot(axs[3],ecfreq05,"w")
axs[3].set_ylabel("f [Hz]")
caxs3.set_ylabel("$\\theta_{k}$")

axs[4], caxs4 = pyrf.plot_spectr(axs[4],dop,yscale="log",clim=[0,1])
pyrf.plot(axs[4],ecfreq,"w")
pyrf.plot(axs[4],ecfreq01,"w")
pyrf.plot(axs[4],ecfreq05,"w")
axs[4].set_ylabel("f [Hz]")
caxs4.set_ylabel("DOP")

axs[5], caxs5 = pyrf.plot_spectr(axs[5],planarity,yscale="log")
pyrf.plot(axs[5],ecfreq,"w")
pyrf.plot(axs[5],ecfreq01,"w")
pyrf.plot(axs[5],ecfreq05,"w")
axs[5].set_ylabel("f [Hz]")
caxs5.set_ylabel("planarity")

axs[6], caxs6 = pyrf.plot_spectr(axs[6],vph,yscale="log",cscale="log")
pyrf.plot(axs[6],ecfreq,"w")
pyrf.plot(axs[6],ecfreq01,"w")
pyrf.plot(axs[6],ecfreq05,"w")
axs[6].set_ylabel("f [Hz]")
caxs6.set_ylabel("E/B"+"\n"+"[m.s$^{-1}$]")

axs[7], caxs7 = pyrf.plot_spectr(axs[7],pfluxz,yscale="log",cmap="seismic",clim=[-1,1])
pyrf.plot(axs[7],ecfreq,"w")
pyrf.plot(axs[7],ecfreq01,"w")
pyrf.plot(axs[7],ecfreq05,"w")
axs[7].set_ylabel("f [Hz]")

caxs7.set_ylabel("$S_{||}/|S|$")

plt.show()