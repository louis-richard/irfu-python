# Plots of B, J, E, JxB electric field, and J.E. Calculates J using
# Curlometer method. 
# Written by L. RICHARD

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


Tint = ["2016-06-07T01:53:44.000","2016-06-07T19:30:34.000"]


ic = np.arange(1,5)

for i in ic:
	exec("Bxyz? = mms.get_data('B_dmpa_dfg_srvy_l2pre',Tint,?)".replace("?",str(i)))
	exec("Bxyz? = pyrf.resample(Bxyz?,Bxyz1)".replace("?",str(i)))

Bxyzav = pyrf.ts_vec_xyz(Bxyz1.time.data,(Bxyz1.data+Bxyz2.data+Bxyz3.data+Bxyz4.data)/4);

# Load electric field
for i in ic:
	exec("Exyz? = mms.get_data('E2d_dsl_edp_fast_l2pre',Tint,?)".replace("?",str(i)))
	exec("Exyz? = pyrf.resample(Exyz?,Exyz1)".replace("?",str(i)))
	#exec("[Exyz?,d?] = pyrf.edb(Exyz?,Bxyz?,15,'E.B=0')".replace("?",str(i))) 			# Removes some wake fields

Exyzav = pyrf.ts_vec_xyz(Exyz1.time.data,(Exyz1.data+Exyz2.data+Exyz3.data+Exyz4.data)/4)


for i in ic:
	#exec("ni? = mms.get_data('Ni_fpi_fast_l2',Tint,?)".replace("?",str(i))) 			# For MP phases
	exec("ni? = mms.get_data('Nhplus_hpca_srvy_l2',Tint,?)".replace("?",str(i))) 		# For 1x phases
	exec("ni? = pyrf.resample(ni?,ni1)".replace("?",str(i)))

ni = pyrf.ts_scalar(ni1.time.data,(ni1.data+ni2.data+ni3.data+ni4.data)/4)
ni = pyrf.resample(ni,Bxyz1)

for i in ic:
	exec("Rxyz? = mms.get_data('R_gse',Tint,?)".replace("?",str(i)))
	exec("Rxyz? = pyrf.resample(Rxyz?,Bxyz1)".replace("?",str(i)))


# Assuming GSE and DMPA are the same coordinate system.
j, divB, jxB, divTshear, divPb = pyrf.c_4_j(Rxyz1,Rxyz2,Rxyz3,Rxyz4,Bxyz1,Bxyz2,Bxyz3,Bxyz4)

divovercurl 		= divB
divovercurl.data 	= np.abs(divB.data)/np.linalg.norm(j)

# Transform current density into field-aligned coordinates
jfac = pyrf.convert_fac(j,Bxyzav,[1,0,0])


# plot
fig, axs = plt.subplots(8,sharex=True,figsize=(6.5,9))
fig.subplots_adjust(bottom=0.1,top=0.95,left=0.15,right=0.85,hspace=0)
pltrf.plot_line(axs[0],Bxyzav)
axs[0].set_ylabel("$B_{DMPA}$"+"\n"+"[nT]")
axs[0].legend(["$B_x$","$B_y$","$B_z$"],frameon=False,ncol=3,loc="upper right")
axs[0].set_ylim([-70,70])
axs[0].text(0.02,0.83,"(a)", transform=axs[0].transAxes)

pltrf.plot_line(axs[1],ni1)
pltrf.plot_line(axs[1],ni2)
pltrf.plot_line(axs[1],ni3)
pltrf.plot_line(axs[1],ni4)
axs[1].set_ylabel("$n_i$"+"\n"+"[cm$^{-3}$]")
axs[1].set_yscale("log")
axs[1].set_ylim([1e-4,1e1])
axs[1].text(0.02,0.83,"(b)", transform=axs[1].transAxes)
axs[1].legend(["MMS1","MMS2","MMS3","MMS4"],frameon=False,ncol=4,loc="upper right")

j.data *= 1e9
pltrf.plot_line(axs[2],j)
axs[2].set_ylabel("$J_{DMPA}$"+"\n"+"[nA.m$^{-2}$]")
axs[2].legend(["$J_x$","$J_y$","$J_z$"],frameon=False,ncol=3,loc="upper right")
axs[2].text(0.02,0.83,"(c)", transform=axs[2].transAxes)

jfac.data *= 1e9
pltrf.plot_line(axs[3],jfac)
axs[3].set_ylabel("$J_{FAC}$"+"\n"+"[nA.m$^{-2}$]")
axs[3].legend(["$J_{\\perp 1}$","$J_{\\perp 2}$","$J_{\\parallel}$"],frameon=False,ncol=3,loc="upper right")
axs[3].text(0.02,0.83,"(d)", transform=axs[3].transAxes)

pltrf.plot_line(axs[4],divovercurl)
axs[4].set_ylabel("$\\frac{|\\nabla . B|}{|\\nabla \\times B|}$")
axs[4].text(0.02,0.83,"(e)", transform=axs[4].transAxes)

pltrf.plot_line(axs[5],Exyzav)
axs[5].set_ylabel("$E_{DSL}$"+"\n"+"[mV.m$^{-1}$]")
axs[5].legend(["$E_x$","$E_y$","$E_z$"],frameon=False,ncol=3,loc="upper right")
axs[5].text(0.02,0.83,"(f)", transform=axs[5].transAxes)

jxB.data /= ni.data[:,np.newaxis]
jxB.data /= 1.6e-19*1000 								# Convert to (mV/m)
jxB.data[np.linalg.norm(jxB.data,axis=1)>100] = np.nan 	# Remove some questionable fields
pltrf.plot_line(axs[6],jxB)
axs[6].set_ylabel("$J \\times B/n_{e}q_{e}$"+"\n"+"[mV.m$^{-1}$]")
axs[6].text(0.02,0.83,"(g)", transform=axs[6].transAxes)

j 		= pyrf.resample(j,Exyzav)
EdotJ 	= pyrf.dot(Exyzav,j)/1000 #J (nA/m^2), E (mV/m), E.J (nW/m^3)

pltrf.plot_line(axs[7],EdotJ)
axs[7].set_ylabel("$E . J$"+"\n"+"[nW.m$^{-3}]$");
axs[7].text(0.02,0.83,"(h)", transform=axs[7].transAxes)

axs[0].set_title("MMS - Current density and fields")
fig.align_ylabels(axs)
axs[-1].set_xlim(Tint)

