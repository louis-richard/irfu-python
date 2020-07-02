# coding: utf-8
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# # Title 	: Example_MMS_EBfields.ipynb
# Author 	: Louis RICHARD\
# E-Mail 	: louisr@irfu.se\
# Created 	: 30-April-20\
# Updated 	: 04-May-20
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


import os
import pyrf
import xarray as xr
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

"""
plt.style.use('bmh')
color 	= [mcd.XKCD_COLORS["xkcd:navy"],\
		 	mcd.XKCD_COLORS["xkcd:khaki"],\
		 	mcd.XKCD_COLORS["xkcd:crimson"],\
		 	mcd.XKCD_COLORS["xkcd:darkgreen"]]

default_cycler = cycler(color=color)
plt.rc('axes',prop_cycle=default_cycler)
"""
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

# Path of MMS data and figures
data_path = "/Volumes/mms"
fig_path = "/Users/louisr/Documents/PhD/Y1/"
#fig_path = "/Users/louisr/Documents/PhD/Y1/EGU/figures_eps"
#fig_path = "/Users/louisr/Documents/PhD/Y1/202003/flapping_20190914/"

# Define time interval
Tint = ["2017-07-17T07:48:30.00","2017-07-17T07:50:20.00"]
#Tint = ["2019-09-14T08:00:00.00","2019-09-14T08:01:00.00"]

# Spacecraft index
ic = 1

#---------------------------------------------------------------------------------------------------------------------
# Load data
#---------------------------------------------------------------------------------------------------------------------
# Background magnetic field from FGM
exec("Bxyz = pyrf.get_data('B_gse_fgm_brst_l2',Tint,?)".replace("?",str(ic)))
# Electric field from EDP
exec("Exyz = pyrf.get_data('E_gse_edp_brst_l2',Tint,?)".replace("?",str(ic)))
# Magnetic field fluctuations from SCM
exec("Bscm = pyrf.get_data('B_gse_scm_brst_l2',Tint,?)".replace("?",str(ic)))
# Number density from FPI
exec("ne = pyrf.get_data('Ne_fpi_brst_l2',Tint,?)".replace("?",str(ic)))


#---------------------------------------------------------------------------------------------------------------------
# Minimum variance analysis
#---------------------------------------------------------------------------------------------------------------------
V = np.array([[-0.32214458,-0.89543338,-0.30728151],\
                [0.81489092,-0.09707433,-0.57142747],\
                [0.48184608,-0.43448318,0.76095251]])


lmnB        = pyrf.new_xyz(Bxyz,V)

#---------------------------------------------------------------------------------------------------------------------
# Convert to field aligned coordinates
#---------------------------------------------------------------------------------------------------------------------
Exyzfac = pyrf.convert_fac(Exyz,Bxyz,[1,0,0])
Bscmfac = pyrf.convert_fac(Bscm,Bxyz,[1,0,0])

#---------------------------------------------------------------------------------------------------------------------
# Filter 
#---------------------------------------------------------------------------------------------------------------------
# Bandpass filter E and B waveforms
fmin = 4
fmax = 1000
Exyzfachf = pyrf.filt(Exyzfac,fmin,0,3)
Exyzfaclf = pyrf.filt(Exyzfac,0,fmin,3)
Bscmfachf = pyrf.filt(Bscmfac,fmin,0,3)
fmin = 0.5

#---------------------------------------------------------------------------------------------------------------------
# Wavelet transforms
#---------------------------------------------------------------------------------------------------------------------
# Wavelet transform field aligned electric field
import time
nf = 100
start = time.time()
Ewavelet = pyrf.wavelet(Exyzfac,nf=nf,f=[fmin,fmax],plot=False)
end = time.time()
print(end - start)

start = time.time()
Bwavelet = pyrf.wavelet(Bscm,nf=nf,f=[fmin,fmax],plot=False)
end = time.time()
print(end - start)


fmin = 4
nc = 100
idx = np.arange(int(nc/2),len(Ewavelet.time)-int(nc/2),step=nc).astype(int)
Ewavelettimes = Ewavelet.time[idx]
Ewaveletx = np.zeros((len(idx),nf))
Ewavelety = np.zeros((len(idx),nf))
Ewaveletz = np.zeros((len(idx),nf))

for ii in range(len(idx)):
    Ewaveletx[ii,:] = np.squeeze(np.nanmean(Ewavelet.x[idx[ii]-int(nc/2)+1:idx[ii]+int(nc/2),:],axis=0))
    Ewavelety[ii,:] = np.squeeze(np.nanmean(Ewavelet.y[idx[ii]-int(nc/2)+1:idx[ii]+int(nc/2),:],axis=0))
    Ewaveletz[ii,:] = np.squeeze(np.nanmean(Ewavelet.z[idx[ii]-int(nc/2)+1:idx[ii]+int(nc/2),:],axis=0))
    
specperpE = xr.DataArray(Ewaveletx+Ewavelety,coords=[Ewavelettimes,Ewavelet.frequency],dims=["time","frequency"])
specparaE = xr.DataArray(Ewaveletz,coords=[Ewavelettimes,Ewavelet.frequency],dims=["time","frequency"])

# Wavelet transform of magnetic field fluctuations
idx = np.arange(int(nc/2),len(Bwavelet.time)-int(nc/2),step=nc).astype(int)
Bwavelettimes = Ewavelet.time[idx]
Bwaveletx = np.zeros((len(idx),nf))
Bwavelety = np.zeros((len(idx),nf))
Bwaveletz = np.zeros((len(idx),nf))

for ii in range(len(idx)):
    Bwaveletx[ii,:] = np.squeeze(np.nanmean(Bwavelet.x[idx[ii]-int(nc/2)+1:idx[ii]+int(nc/2),:],axis=0))
    Bwavelety[ii,:] = np.squeeze(np.nanmean(Bwavelet.y[idx[ii]-int(nc/2)+1:idx[ii]+int(nc/2),:],axis=0))
    Bwaveletz[ii,:] = np.squeeze(np.nanmean(Bwavelet.z[idx[ii]-int(nc/2)+1:idx[ii]+int(nc/2),:],axis=0))

specB = xr.DataArray(Bwaveletx+Bwavelety+Bwaveletz,coords=[Bwavelettimes,Bwavelet.frequency],\
						dims=["time","frequency"])

del(Ewavelet,Ewavelettimes,Ewaveletx,Ewavelety,Ewaveletz)
del(Bwavelet,Bwavelettimes,Bwaveletx,Bwavelety,Bwaveletz)


#---------------------------------------------------------------------------------------------------------------------
# Compute plasma parameters
#---------------------------------------------------------------------------------------------------------------------
pparam = pyrf.plasma_calc(Bxyz,ne,ne,ne,ne)


#---------------------------------------------------------------------------------------------------------------------
# Plot
#---------------------------------------------------------------------------------------------------------------------
cmap = "jet"
fig, axs = plt.subplots(6,sharex=True,figsize=(6.5,9))
fig.subplots_adjust(bottom=0.05,top=0.95,left=0.15,right=0.85,hspace=0.)
pyrf.plot(axs[0],lmnB)
axs[0].legend(["$B_x$","$B_y$","$B_z$"],ncol=3,frameon=False,loc="upper right")
axs[0].set_ylabel("$B$ [nT]")

pyrf.plot(axs[1],Exyzfaclf)
axs[1].legend(["$E_{\\perp 1}$","$E_{\\perp 2}$","$E_{\\parallel}$"],ncol=3,frameon=False,loc="upper right")
axs[1].set_ylabel("$E$ [mV.m$^{-1}$]")
axs[1].text(0.02,0.83,"(b)", transform=axs[1].transAxes)

pyrf.plot(axs[2],Exyzfachf)
axs[2].legend(["$E_{\\perp 1}$","$E_{\\perp 2}$","$E_{\\parallel}$"],ncol=3,frameon=False,loc="upper right")
axs[2].set_ylabel("$E$ [mV.m$^{-1}$]")
axs[2].text(0.02,0.15,"$f > ${:2.1f} Hz".format(fmin), transform=axs[2].transAxes)

axs[3], caxs3 = pyrf.plot_spectr(axs[3],specperpE,cscale="log",yscale="log",cmap=cmap)
pyrf.plot(axs[3],pparam.Flh)
pyrf.plot(axs[3],pparam.Fce)
pyrf.plot(axs[3],pparam.Fpp)
axs[3].set_ylabel("$f$ [Hz]")
caxs3.set_ylabel("$E_{\\perp}^2$ " + "\n" + "[mV$^2$.m$^{-2}$.Hz$^{-1}$]")
axs[3].legend(["$f_{lh}$","$f_{ce}$","$f_{pi}$"],ncol=3,loc="upper right",frameon=True)

axs[4], caxs4 = pyrf.plot_spectr(axs[4],specparaE,cscale="log",yscale="log",cmap=cmap)
pyrf.plot(axs[4],pparam.Flh)
pyrf.plot(axs[4],pparam.Fce)
pyrf.plot(axs[4],pparam.Fpp)
axs[4].set_ylabel("$f$ [Hz]")
caxs4.set_ylabel("$E_{||}^2$ " + "\n" + "[mV$^2$.m$^{-2}$.Hz$^{-1}$]")
axs[4].legend(["$f_{lh}$","$f_{ce}$","$f_{pi}$"],ncol=3,loc="upper right",frameon=True)


axs[5], caxs5 = pyrf.plot_spectr(axs[5],specB,cscale="log",yscale="log",cmap=cmap)
pyrf.plot(axs[5],pparam.Flh)
pyrf.plot(axs[5],pparam.Fce)
pyrf.plot(axs[5],pparam.Fpp)
axs[5].set_ylabel("$f$ [Hz]")
caxs5.set_ylabel("$B^2$ " + "\n" + "[nT$^2$.Hz$^{-1}$]")
axs[5].legend(["$f_{lh}$","$f_{ce}$","$f_{pi}$"],ncol=3,loc="upper right",frameon=True)

axs[0].text(0.02,0.83,"(a)", transform=axs[0].transAxes)
axs[1].text(0.02,0.83,"(b)", transform=axs[1].transAxes)
axs[2].text(0.02,0.83,"(c)", transform=axs[2].transAxes)
axs[3].text(0.02,0.83,"(d)", transform=axs[3].transAxes)
axs[4].text(0.02,0.83,"(e)", transform=axs[4].transAxes)
axs[5].text(0.02,0.83,"(f)", transform=axs[5].transAxes)

fig.align_ylabels(axs)
axs[-1].set_xlabel("2019-09-14 UTC")
#fig.align_ylabels([caxs3,caxs4,caxs5])

fig_name = "_".join([pyrf.fname(Tint,3),"EBfields.png"])
fig.savefig(os.path.join(fig_path,fig_name),format="png")

#fig_name = "_".join([pyrf.fname(Tint,3),"EBfields.pdf"])
#fig.savefig(os.path.join(fig_path,fig_name),format="pdf")
plt.show()
#---------------------------------------------------------------------------------------------------------------------
# End
#---------------------------------------------------------------------------------------------------------------------