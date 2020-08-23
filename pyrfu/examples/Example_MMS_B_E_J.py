# Plots of B, J, E, JxB electric field, and J.E. Calculates J using
# Curlometer method. 
# Written by L. RICHARD
import numpy as np
import matplotlib.pyplot as plt

from pyrfu.mms import get_data
from pyrfu.pyrf import avg_4sc, resample, c_4_j, convert_fac, dot
from pyrfu.plot import plot_line


def main(Tint):
	# Load magnetic field
	B 	= [get_data("B_dmpa_dfg_srvy_l2",Tint,ic) for ic in range(1,5)]
	Bav = avg_4sc(B)

	# Load electric field
	E 	= [get_data("E2d_dsl_edp_fast_l2",Tint,ic) for ic in range(1,5)]
	Eav = avg_4sc(E)

	# Load H+ number density
	N_i = [get_data("hplus_hpca_srvy_l2",Tint,ic) for ic in range(1,5)]
	n_i = avg_4sc(N_i)
	n_i = resample(n_i,B[0])

	# Load spacecraft position
	R 	= [get_data("R_gse",Tint,ic) for ic in range(1,5)]

	# Assuming GSE and DMPA are the same coordinate system.
	j, divB, Bav, jxB, divTshear, divPb = c_4_j(R,B)
	
	divovercurl 		= divB
	divovercurl.data 	= np.abs(divB.data)/np.linalg.norm(j)

	jxB.data /= n_i.data[:,np.newaxis]
	jxB.data /= 1.6e-19*1000 								# Convert to (mV/m)

	jxB.data[np.linalg.norm(jxB.data,axis=1)>100] = np.nan 	# Remove some questionable fields

	# Transform current density into field-aligned coordinates
	jfac = convert_fac(j,Bxyzav,[1,0,0])

	# Compute dissipation terms
	j 		= resample(j,Exyzav)
	EdotJ 	= dot(Exyzav,j)/1000 #J (nA/m^2), E (mV/m), E.J (nW/m^3)


	# plot
	fig, axs = plt.subplots(8,sharex=True,figsize=(6.5,9))
	fig.subplots_adjust(bottom=0.1,top=0.95,left=0.15,right=0.85,hspace=0)

	plot_line(axs[0],Bav)
	axs[0].set_ylabel("$B_{DMPA}$"+"\n"+"[nT]")
	axs[0].legend(["$B_x$","$B_y$","$B_z$"],frameon=False,ncol=3,loc="upper right")
	axs[0].set_ylim([-70,70])
	axs[0].text(0.02,0.83,"(a)", transform=axs[0].transAxes)

	for n in N_i: plot_line(axs[1],n)
	axs[1].set_ylabel("$n_i$"+"\n"+"[cm$^{-3}$]")
	axs[1].set_yscale("log")
	axs[1].set_ylim([1e-4,1e1])
	axs[1].text(0.02,0.83,"(b)", transform=axs[1].transAxes)
	axs[1].legend(["MMS1","MMS2","MMS3","MMS4"],frameon=False,ncol=4,loc="upper right")

	j.data *= 1e9
	plot_line(axs[2],j)
	axs[2].set_ylabel("$J_{DMPA}$"+"\n"+"[nA.m$^{-2}$]")
	axs[2].legend(["$J_x$","$J_y$","$J_z$"],frameon=False,ncol=3,loc="upper right")
	axs[2].text(0.02,0.83,"(c)", transform=axs[2].transAxes)

	jfac.data *= 1e9
	plot_line(axs[3],jfac)
	axs[3].set_ylabel("$J_{FAC}$"+"\n"+"[nA.m$^{-2}$]")
	axs[3].legend(["$J_{\\perp 1}$","$J_{\\perp 2}$","$J_{\\parallel}$"],frameon=False,ncol=3,loc="upper right")
	axs[3].text(0.02,0.83,"(d)", transform=axs[3].transAxes)

	plot_line(axs[4],divovercurl)
	axs[4].set_ylabel("$\\frac{|\\nabla . B|}{|\\nabla \\times B|}$")
	axs[4].text(0.02,0.83,"(e)", transform=axs[4].transAxes)

	plot_line(axs[5],Exyzav)
	axs[5].set_ylabel("$E_{DSL}$"+"\n"+"[mV.m$^{-1}$]")
	axs[5].legend(["$E_x$","$E_y$","$E_z$"],frameon=False,ncol=3,loc="upper right")
	axs[5].text(0.02,0.83,"(f)", transform=axs[5].transAxes)


	pltrf.plot_line(axs[6],jxB)
	axs[6].set_ylabel("$J \\times B/n_{e}q_{e}$"+"\n"+"[mV.m$^{-1}$]")
	axs[6].text(0.02,0.83,"(g)", transform=axs[6].transAxes)



	pltrf.plot_line(axs[7],EdotJ)
	axs[7].set_ylabel("$E . J$"+"\n"+"[nW.m$^{-3}]$");
	axs[7].text(0.02,0.83,"(h)", transform=axs[7].transAxes)

	axs[0].set_title("MMS - Current density and fields")
	fig.align_ylabels(axs)
	axs[-1].set_xlim(Tint)



if __name__ == "__main__":
	Tint = ["2016-06-07T01:53:44.000","2016-06-07T19:30:34.000"]

	main(Tint)