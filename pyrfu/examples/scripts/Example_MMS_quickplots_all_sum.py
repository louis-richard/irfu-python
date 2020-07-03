# coding: utf-8
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# # Title 	: Example_MMS_quickplots_all_sum.py
# Author 	: Louis RICHARD
# E-Mail 	: louisr@irfu.se
# Created 	: 05-Jun-20
# Updated 	: 05-Jun-20
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

from pyrfu import pyrf
from pyrfu import mms
from pyrfu import plot as pltrf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib._color_data as mcd
from astropy import constants

date_form = mdates.DateFormatter("%H:%M:%S")
sns.set()
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8,'axes.grid': True,\
                        'font.family': ['sans-serif']})

sns.set_context("paper")
plt.rc('lines', linewidth=0.8)
color = ["k","b","r","g","c","m","y"]
default_cycler = cycler(color=color)
plt.rc('axes',prop_cycle=default_cycler)
plt.close("all")


ic = 1 

Tint = ["2019-09-14T06:00:00.000","2019-09-14T08:00:00.000"]

tmmode = "fast"

if tmmode == "fast":
	btmmode = "srvy"

# Magnetic field
Bxyz = pyrf.get_data("B_gse_fgm_srvy_l2",Tint,ic)
# Density energy flux
iEnflux = pyrf.get_data("Enfluxi_fpi_fast_l2",Tint,ic)
eEnflux = pyrf.get_data("Enfluxe_fpi_fast_l2",Tint,ic)
# Number density
Ni = pyrf.get_data("Ni_fpi_fast_l2",Tint,ic)
Ne = pyrf.get_data("Ne_fpi_fast_l2",Tint,ic)
# Spacecraft potential
scPot = pyrf.get_data("V_edp_fast_l2",Tint,ic)
# Ion bulk velocity
Vi = pyrf.get_data("Vi_gse_fpi_fast_l2",Tint,ic)
# Electric field
Exyz = pyrf.get_data("E_gse_edp_fast_l2",Tint,ic)
# PSD of magnetic field
Bpsd = pyrf.get_data("Bpsd_dsp_fast_l2",Tint,ic)
# PSD of electric field
Epsd = pyrf.get_data("Epsd_dsp_fast_l2",Tint,ic)


# Compute ExB drift 

#ExB = 1e-3*pyrf.cross(1e-3*Exyz,1e-9*Bxyz)/(np.abs(1e-9*pyrf.resample(Bxyz,Exyz))**2)
#print(ExB)
# Compute parallel and perpandicular ion bulk velocity
Vipara, Viperp, alphai = pyrf.dec_parperp(Vi,Bxyz)


fig, axs = plt.subplots(10,sharex=True,figsize=(6.5,9))
fig.subplots_adjust(bottom=0.1,top=0.95,left=0.15,right=0.85,hspace=0)
axs[0] = pyrf.plot(axs[0],Bxyz)

axs[1], caxs1 = pyrf.plot_spectr(axs[1],iEnflux,yscale="log",cscale="log",cmap="viridis")

axs[2], caxs1 = pyrf.plot_spectr(axs[2],eEnflux,yscale="log",cscale="log",cmap="viridis")

pyrf.plot(axs[3],Ni)
pyrf.plot(axs[3],Ne)

pyrf.plot(axs[4],-np.log(scPot))
pyrf.plot(axs[5],Vi)

#pyrf.plot(axs[5],ExB[:,2])
pyrf.plot(axs[6],Viperp[:,2])

pyrf.plot(axs[7],Exyz)


axs[8], caxs8 = pyrf.plot_spectr(axs[8],Bpsd,yscale="log",cscale="log",cmap="viridis")

axs[9], caxs9 = pyrf.plot_spectr(axs[9],Epsd,yscale="log",cscale="log",cmap="viridis")

plt.show()