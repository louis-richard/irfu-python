# coding: utf-8
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Title     : Example_MMS_EDP_vs_FPI_vs_HPCA_brst.py
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
fig_path = "/Users/louisr/Documents/PhD/Y1/EGU/figures"
data_path = "/Volumes/mms"

# Spacecraft index
mmsId = 4

# Define time interval
Tint = ["2019-09-14T07:54:00","2019-09-14T08:11:00"]

#---------------------------------------------------------------------------------------------------------------------
# Load data
#---------------------------------------------------------------------------------------------------------------------
# FGM/DFG
B_dmpa_fgm_srvy = pyRF.get_data("B_dmpa_fgm_srvy_l2",Tint,mmsId)
if B_dmpa_fgm_srvy is None:
    print("loading l2pre DFG\n")
    B_dmpa_fgm_srvy = pyRF.get_data("B_dmpa_dfg_srvy_l2pre",Tint,mmsId)
    if B_dmpa_fgm_srvy is None:
        print("loading QL DFG\n")
        B_dmpa_fgm_srvy = pyRF.get_data("B_dmpa_dfg_srvy_ql",Tint,mmsId)

# EDP
E_dsl_edp = pyRF.get_data("E_dsl_edp_brst_l2",Tint,mmsId)
if E_dsl_edp is None:
    print("loading QL DCE\n")
    E_dsl_edp = pyRF.get_data("E_dsl_edp_brst_ql",Tint,mmsId)

# In spin plane electric field
E2d_dsl_edp = pyRF.get_data("E2d_dsl_edp_brst_l2pre",Tint,mmsId)
if E2d_dsl_edp is None:
    print("loading QL DCE2d\n")
    E2d_dsl_edp = pyRF.get_data("E2d_dsl_edp_brst_ql",Tint,mmsId)

# ADP
E_adp_edp = pyRF.get_data("E_ssc_edp_brst_l1b",Tint,mmsId)
E_adp_edp = -E_adp_edp[:,2]*1.5

# FPI
Vi_dbcs_fpi = pyRF.get_data("Vi_dbcs_fpi_brst_l2",Tint,mmsId)
Ve_dbcs_fpi = pyRF.get_data("Ve_dbcs_fpi_brst_l2",Tint,mmsId)

# HPCA
Vhplus_dbcs_hpca = pyRF.get_data("Vhplus_dbcs_hpca_brst_l2",Tint,mmsId)
if Vhplus_dbcs_hpca is None:
    Vhplus_dbcs_hpca = pyRF.get_data("Vhplus_dbcs_hpca_brst_l1b",Tint,mmsId)


#---------------------------------------------------------------------------------------------------------------------
# Decompose parallel and perpandicular components
#---------------------------------------------------------------------------------------------------------------------
Vi_para, Vi_perp, alpha         = pyRF.dec_parperp(Vi_dbcs_fpi,B_dmpa_fgm_srvy)
Ve_para, Ve_perp, alpha         = pyRF.dec_parperp(Ve_dbcs_fpi,B_dmpa_fgm_srvy)
E_para, E_perp, alpha           = pyRF.dec_parperp(E_dsl_edp,B_dmpa_fgm_srvy)
Vhplus_para, Vhplus_perp, alpha = pyRF.dec_parperp(Vhplus_dbcs_hpca,B_dmpa_fgm_srvy)


#---------------------------------------------------------------------------------------------------------------------
# Compute velocity from electric fields and electric field from velocities
#---------------------------------------------------------------------------------------------------------------------
VExB        = pyRF.e_vxb(E_dsl_edp,B_dmpa_fgm_srvy,-1)
VE2dxB      = pyRF.e_vxb(E2d_dsl_edp,B_dmpa_fgm_srvy,-1)
EVixB       = pyRF.e_vxb(Vi_dbcs_fpi,pyRF.resample(B_dmpa_fgm_srvy,Vi_dbcs_fpi))
EVexB       = pyRF.e_vxb(Ve_dbcs_fpi,pyRF.resample(B_dmpa_fgm_srvy,Ve_dbcs_fpi))
EVphlusxB   = pyRF.e_vxb(Vhplus_dbcs_hpca,pyRF.resample(B_dmpa_fgm_srvy,Vhplus_dbcs_hpca))


#---------------------------------------------------------------------------------------------------------------------
# plot
#---------------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(3,sharex=True,figsize=(16,9))
fig.subplots_adjust(bottom=.1,top=.9,left=.1,right=.9,hspace=0.05)
for i in range(3):
    pyRF.plot(axs[i],VExB[:,i])
    pyRF.plot(axs[i],VE2dxB[:,i])
    pyRF.plot(axs[i],Ve_perp[:,i])
    pyRF.plot(axs[i],Vi_perp[:,i])
    pyRF.plot(axs[i],Vhplus_perp[:,i])
    axs[i].legend(["VExB","VE2dxB","$V_{e,\\perp}$",\
                    "$V_{i,\\perp}$","$V_{H+,\\perp}$"],ncol=5,frameon=False,loc="upper right")
    
axs[0].set_ylabel("$V_x$ DSL [km.s$^{-1}$]")
axs[1].set_ylabel("$V_y$ DSL [km.s$^{-1}$]")
axs[2].set_ylabel("$V_z$ DSL [km.s$^{-1}$]")

axs[-1].set_xlim(Tint)
axs[0].set_title("MMS{:d}".format(mmsId))
axs[-1].set_xlabel("{} UTC".format(Tint[0][:10]))

fig_name = "_".join([pyRF.fname(Tint,3),"VExB_EDP_vs_FPI_vs_HPCA_brst.png".format(edpMode,fpiMode)])
fig.savefig(os.path.join(fig_path,fig_name),format="png")

fig, axs = plt.subplots(3,sharex=True,figsize=(16,9))
fig.subplots_adjust(bottom=.1,top=.9,left=.1,right=.9,hspace=0.05)
for i in range(3):
    pyRF.plot(axs[i],E2d_dsl_edp[:,i])
    pyRF.plot(axs[i],E_dsl_edp[:,i])
    pyRF.plot(axs[i],E_perp[:,i])
    pyRF.plot(axs[i],EVexB[:,i])
    pyRF.plot(axs[i],EVixB[:,i])
    pyRF.plot(axs[i],EVphlusxB[:,i])
    
pyRF.plot(axs[2],E_adp_edp)
axs[0].legend(["$E2d$","$E$","$E_\\perp$","$V_{e}\\times B$",\
                "$V_{i}\\times B$","$V_{H+}\\times B$"],ncol=6,frameon=False,loc="upper right")
axs[1].legend(["$E2d$","$E$","$E_\\perp$","$V_{e}\\times B$",\
                "$V_{i}\\times B$","$V_{H+}\\times B$"],ncol=6,frameon=False,loc="upper right")
axs[2].legend(["$E2d$","$E$","$E_\\perp$","$V_{e}\\times B$",
                "$V_{i}\\times B$","$V_{H+}\\times B$","$E$ adp"],ncol=7,frameon=False,loc="upper right")

axs[0].set_ylabel("$E_x$ DSL [mV.m$^{-1}$]")
axs[1].set_ylabel("$E_y$ DSL [mV.m$^{-1}$]")
axs[2].set_ylabel("$E_z$ DSL [mV.m$^{-1}$]")

axs[-1].set_xlim(Tint)
axs[0].set_title("MMS{:d}".format(mmsId))
axs[-1].set_xlabel("{} UTC".format(Tint[0][:10]))

fig_name = "_".join([pyRF.fname(Tint,3),"E_EDP_vs_FPI_vs_HPCA_brst.png".format(edpMode,fpiMode)])
fig.savefig(os.path.join(fig_path,fig_name),format="png")

plt.show()
