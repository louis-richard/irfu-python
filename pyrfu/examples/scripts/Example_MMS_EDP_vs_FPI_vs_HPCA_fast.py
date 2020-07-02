# coding: utf-8
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Title     : Example_MMS_EDP_vs_FPI_vs_HPCA_fast.py
# Author    : Louis RICHARD
# E-Mail    : louisr@irfu.se
# Created   : 06-May-20
# Updated   : 06-May-20
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Notes :
# 
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

import pyRF
import os
import numpy as np
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
data_path = "/Volumes/mms"
fig_path = "/Users/louisr/Documents/PhD/Y1/EGU/figures"

# Spacecraft index
mmsId   = 1

# Define time interval
#Tint    = ["2016-08-10T09:50:00","2016-08-10T10:15:00"]
Tint    = ["2019-09-14T07:54:00","2019-09-14T08:11:00"]

# Define time modes
fpiMode = "fast" # alternative fpiMode = 'brst'
edpMode = "fast" # alternative edpMode = 'brst'

# FGM & EDP
if edpMode == "fast":
  fgmMode = "srvy"
elif edpMode == "brst":
  fgmMode = "brst"
else :
  print("unrecognized mode (FAST/BRST)\n")

#---------------------------------------------------------------------------------------------------------------------
# Load data
#---------------------------------------------------------------------------------------------------------------------
# FGM/DFG
B_dmpa_fgm_srvy_l2  = pyRF.get_data("B_dmpa_fgm_{}_l2".format(fgmMode),Tint,mmsId)
E_dsl_edp_l2        = pyRF.get_data("E_dsl_edp_{}_l2".format(edpMode),Tint,mmsId)
E2d_dsl_edp_l2pre   = pyRF.get_data("E2d_dsl_edp_{}_l2pre".format(edpMode),Tint,mmsId)
E_adp_edp           = pyRF.get_data("E_ssc_edp_{}_l1b".format(edpMode),Tint,mmsId)
E_adp_edp           = -E_adp_edp[:,2]*1.5

# FPI
fpiSuf = "fpi_{}_l2".format(fpiMode)
Vi_dbcs_fpi = pyRF.get_data("Vi_dbcs_{}".format(fpiSuf),Tint,mmsId)
Ve_dbcs_fpi = pyRF.get_data("Ve_dbcs_{}".format(fpiSuf),Tint,mmsId)
Ne_fpi      = pyRF.get_data("Ne_{}".format(fpiSuf),Tint,mmsId)

Ve_dbcs_fpi.data[Ne_fpi.data < 0.06,:] = np.nan

# HPCA
Vhplus_dbcs_hpca = pyRF.get_data("Vhplus_dbcs_hpca_srvy_l2",Tint,mmsId)
if Vhplus_dbcs_hpca is None:
  Vhplus_dbcs_hpca = pyRF.get_data("Vhplus_dbcs_hpca_srvy_l1b",Tint,mmsId)


# correct Ez in E2d 
# XXX: this should be undone later 
E2d_dsl_edp_l2pre, d = pyRF.edb(E2d_dsl_edp_l2pre,B_dmpa_fgm_srvy_l2,10,"Eperp+NaN")

# Comp VxB
[Vi_para, Vi_perp, alpha] = pyRF.dec_parperp(Vi_dbcs_fpi,B_dmpa_fgm_srvy_l2)
[Ve_para, Ve_perp, alpha] = pyRF.dec_parperp(Ve_dbcs_fpi,B_dmpa_fgm_srvy_l2)

VExB        = pyRF.e_vxb(E_dsl_edp_l2,B_dmpa_fgm_srvy_l2,-1);
VExB_l2pre  = pyRF.e_vxb(E2d_dsl_edp_l2pre,B_dmpa_fgm_srvy_l2,-1);
EVixB       = pyRF.e_vxb(Vi_dbcs_fpi,pyRF.resample(B_dmpa_fgm_srvy_l2,Vi_dbcs_fpi))
EVexB       = pyRF.e_vxb(Ve_dbcs_fpi,pyRF.resample(B_dmpa_fgm_srvy_l2,Ve_dbcs_fpi))

if Vhplus_dbcs_hpca is None:
    Vhplus_perp = None
    EVphlusxB = None
else :
    [Vhplus_para, Vhplus_perp, alpha] = pyRF.dec_parperp(Vhplus_dbcs_hpca,B_dmpa_fgm_srvy_l2,)
    EVhplusxB = pyRF.e_vxb(Vhplus_dbcs_hpca,pyRF.resample(B_dmpa_fgm_srvy_l2,Vhplus_dbcs_hpca))


#---------------------------------------------------------------------------------------------------------------------
# plot
#---------------------------------------------------------------------------------------------------------------------
# E plot
fig, axs = plt.subplots(4,sharex=True,figsize=(16,9))
fig.subplots_adjust(bottom=.1,top=.9,left=.1,right=.9,hspace=0.)
pyRF.plot(axs[0],B_dmpa_fgm_srvy_l2)
axs[0].set_title("MMS{:d}".format(mmsId))
axs[0].set_ylabel("$B$ DSL [nT]")
axs[0].legend(["X","Y","Z"],ncol=3,frameon=False,loc="upper right")

for i in range(3):
  pyRF.plot(axs[i+1],E2d_dsl_edp_l2pre[:,i])
  pyRF.plot(axs[i+1],E_dsl_edp_l2[:,i])
  pyRF.plot(axs[i+1],EVexB[:,i])
  pyRF.plot(axs[i+1],EVixB[:,i])
  pyRF.plot(axs[i+1],EVhplusxB[:,i])
  axs[i+1].legend(["E L2pre","E l2","$V_{e}\\times B$","$V_{i}\\times B$","$V_{H+}\\times B$"],\
          ncol=5,frameon=False,loc="upper right")

pyRF.plot(axs[-1],E_adp_edp)
axs[-1].legend(["E L2pre","E l2","$V_{e}\\times B$","$V_{i}\\times B$","$V_{H+}\\times B$","E ADP"],\
          ncol=6,frameon=False,loc="upper right")

axs[1].set_ylabel("$E_x$ DSL [mV.m$^{-1}$]")
axs[2].set_ylabel("$E_x$ DSL [mV.m$^{-1}$]")
axs[3].set_ylabel("$E_x$ DSL [mV.m$^{-1}$]")

fig.align_ylabels(axs)
axs[-1].set_xlabel("{} UTC".format(Tint[0][:10]))
axs[-1].set_xlim(Tint)

fig_name = "_".join([pyRF.fname(Tint,3),"E_EDP_{}_vs_FPI_{}_vs_HPCA_fast.png".format(edpMode,fpiMode)])
fig.savefig(os.path.join(fig_path,fig_name),format="png")

# V plot
fig, axs = plt.subplots(4,sharex=True,figsize=(16,9))
fig.subplots_adjust(bottom=.1,top=.9,left=.1,right=.9,hspace=0.)
pyRF.plot(axs[0],B_dmpa_fgm_srvy_l2)
axs[0].set_title("MMS{:d}".format(mmsId))
axs[0].set_ylabel("$B$ DSL [nT]")
axs[0].legend(["X","Y","Z"],ncol=3,frameon=False,loc="upper right")

for i in range(3):
  pyRF.plot(axs[i+1],VExB_l2pre[:,i])
  pyRF.plot(axs[i+1],VExB[:,i])
  pyRF.plot(axs[i+1],Ve_perp[:,i])
  pyRF.plot(axs[i+1],Vi_perp[:,i])
  pyRF.plot(axs[i+1],Vhplus_perp[:,i])
  axs[i+1].legend(["VExE L2pre","VExE","$V_{e,\\perp}$","$V_{i,\\perp}$","$V_{H+,\\perp}$"],\
          ncol=5,frameon=False,loc="upper right")

axs[1].set_ylabel("$V_x$ DSL [km.s$^{-1}$]")
axs[2].set_ylabel("$V_y$ DSL [km.s$^{-1}$]")
axs[3].set_ylabel("$V_z$ DSL [km.s$^{-1}$]")

fig.align_ylabels(axs)
axs[-1].set_xlabel("{} UTC".format(Tint[0][:10]))
axs[-1].set_xlim(Tint)

fig_name = "_".join([pyRF.fname(Tint,3),"VExB_EDP_{}_vs_FPI_{}_vs_HPCA_fast.png".format(edpMode,fpiMode)])
fig.savefig(os.path.join(fig_path,fig_name),format="png")


plt.show()