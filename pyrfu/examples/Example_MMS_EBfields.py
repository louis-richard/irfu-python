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

import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


from pyrfu.mms import get_data
from pyrfu.pyrf import convert_fac, filt, wavelet
from pyrfu.plot import plot_line, plot_spectr

# Define time interval
Tint = ["2017-07-17T07:48:30.00","2017-07-17T07:50:20.00"]
#Tint = ["2017-07-18T13:03:34.00","2017-07-18T13:07:00.00"]
#Tint = ["2019-09-14T08:00:00.00","2019-09-14T08:01:00.00"]


def ccwt(S=None,nf=100,f=[5e-1,1e3],plot=False,nc=100):
    """
    Compressed wavelet transform with average over nc time stamps
    """
    Swavelet = wavelet(S,nf=nf,f=f,plot=False)

    idx = np.arange(int(nc/2),len(Swavelet.time)-int(nc/2),step=nc).astype(int)
    Swavelettimes = Swavelet.time[idx]
    Swaveletx = np.zeros((len(idx),nf))
    Swavelety = np.zeros((len(idx),nf))
    Swaveletz = np.zeros((len(idx),nf))

    for ii in range(len(idx)):
        Swaveletx[ii,:] = np.squeeze(np.nanmean(Swavelet.x[idx[ii]-int(nc/2)+1:idx[ii]+int(nc/2),:],axis=0))
        Swavelety[ii,:] = np.squeeze(np.nanmean(Swavelet.y[idx[ii]-int(nc/2)+1:idx[ii]+int(nc/2),:],axis=0))
        Swaveletz[ii,:] = np.squeeze(np.nanmean(Swavelet.z[idx[ii]-int(nc/2)+1:idx[ii]+int(nc/2),:],axis=0))

    Swaveletx = xr.DataArray(Swaveletx,coords=[Swavelettimes,Swavelet.frequency],dims=["time","frequency"])

    return (Swaveletx,Swavelety,Swaveletz)


def plot(axs,Bxyz,Exyzfaclf,Exyzfachf,specperpE,specparaE,specB,pparam):
    #---------------------------------------------------------------------------------------------------------------------
    # Plot
    #---------------------------------------------------------------------------------------------------------------------
    
    plot_line(axs[0],Bxyz)
    axs[0].legend(["$B_x$","$B_y$","$B_z$"],ncol=3,frameon=False,loc="upper right")
    axs[0].set_ylabel("$B$ [nT]")

    plot_line(axs[1],Exyzfaclf)
    axs[1].legend(["$E_{\\perp 1}$","$E_{\\perp 2}$","$E_{\\parallel}$"],ncol=3,frameon=False,loc="upper right")
    axs[1].set_ylabel("$E$ [mV.m$^{-1}$]")
    axs[1].text(0.02,0.83,"(b)", transform=axs[1].transAxes)

    plot_line(axs[2],Exyzfachf)
    axs[2].legend(["$E_{\\perp 1}$","$E_{\\perp 2}$","$E_{\\parallel}$"],ncol=3,frameon=False,loc="upper right")
    axs[2].set_ylabel("$E$ [mV.m$^{-1}$]")
    axs[2].text(0.02,0.15,"$f > ${:2.1f} Hz".format(fmin), transform=axs[2].transAxes)

    axs[3], caxs3 = plot_spectr(axs[3],specperpE,cscale="log",yscale="log")
    plot_line(axs[3],pparam.Flh)
    plot_line(axs[3],pparam.Fce)
    plot_line(axs[3],pparam.Fpp)
    axs[3].set_ylabel("$f$ [Hz]")
    caxs3.set_ylabel("$E_{\\perp}^2$ " + "\n" + "[mV$^2$.m$^{-2}$.Hz$^{-1}$]")
    axs[3].legend(["$f_{lh}$","$f_{ce}$","$f_{pi}$"],ncol=3,loc="upper right",frameon=True)

    axs[4], caxs4 = plot_spectr(axs[4],specparaE,cscale="log",yscale="log")
    plot_line(axs[4],pparam.Flh)
    plot_line(axs[4],pparam.Fce)
    plot_line(axs[4],pparam.Fpp)
    axs[4].set_ylabel("$f$ [Hz]")
    caxs4.set_ylabel("$E_{||}^2$ " + "\n" + "[mV$^2$.m$^{-2}$.Hz$^{-1}$]")
    axs[4].legend(["$f_{lh}$","$f_{ce}$","$f_{pi}$"],ncol=3,loc="upper right",frameon=True)


    axs[5], caxs5 = plot_spectr(axs[5],specB,cscale="log",yscale="log")
    plot_line(axs[5],pparam.Flh)
    plot_line(axs[5],pparam.Fce)
    plot_line(axs[5],pparam.Fpp)
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

    #fig_name = "_".join([pyrf.fname(Tint,3),"EBfields.png"])
    #fig.savefig(os.path.join(fig_path,fig_name),format="png")

    #fig_name = "_".join([pyrf.fname(Tint,3),"EBfields.pdf"])
    #fig.savefig(os.path.join(fig_path,fig_name),format="pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tstart",type=str,default="2017-07-17T07:48:30.00")
    parser.add_argument("--tstop",type=str,default="2017-07-17T07:50:20.00")
    parser.add_argument("--tmmode",type=str,default="brst")
    parser.add_argument("--fmin",type=float,default=4)
    parser.add_argument("--mmsId",type=int,default=1)
    parser.add_argument("--save",type=str,default="")
    
    args = parser.parse_args()

    Tint = [args.tstart,args.tstop]

    #-----------------------------------------------------------------------------------------------------
    # Load data
    #-----------------------------------------------------------------------------------------------------
    # Background magnetic field from FGM
    Bxyz = get_data("B_gse_fgm_brst_l2",Tint,args.mmsId)
    # Electric field from EDP
    Exyz = get_data("E_gse_edp_brst_l2",Tint,args.mmsId)
    # Magnetic field fluctuations from SCM
    Bscm = get_data("B_gse_scm_brst_l2",Tint,args.mmsId)
    # Number density from FPI
    n_e = get_data("Ne_fpi_brst_l2",Tint,args.mmsId)

    #-----------------------------------------------------------------------------------------------------
    # Convert to field aligned coordinates
    #-----------------------------------------------------------------------------------------------------
    Exyzfac = convert_fac(Exyz,Bxyz,[1,0,0])
    Bscmfac = convert_fac(Bscm,Bxyz,[1,0,0])

    #-----------------------------------------------------------------------------------------------------
    # Filter 
    #-----------------------------------------------------------------------------------------------------
    # Bandpass filter E and B waveforms
    Exyzfachf = filt(Exyzfac,args.fmin,0,3)
    Exyzfaclf = filt(Exyzfac,0,args.fmin,3)
    Bscmfachf = filt(Bscmfac,args.fmin,0,3)
    #-----------------------------------------------------------------------------------------------------
    # Wavelet transforms
    #-----------------------------------------------------------------------------------------------------
    # Wavelet transform field aligned electric field
    nf, nc      = [100, 100]
    fmin, fmax  = [0.5, 1000]

    Ewaveletx, Ewavelety, Ewaveletz = ccwt(S=Exyzfac,nf=nf,f=[fmin,fmax],plot=False,nc=nc)

    Bwaveletx, Bwavelety, Bwaveletz = ccwt(S=Bscm,nf=nf,f=[fmin,fmax],plot=False,nc=nc)

    specperpE   = xr.DataArray(Ewaveletx+Ewavelety,coords=[Ewavelettimes,Ewavelet.frequency],\
                                dims=["time","frequency"])

    specparaE   = xr.DataArray(Ewaveletz,coords=[Ewavelettimes,Ewavelet.frequency],\
                                dims=["time","frequency"])

    specB       = xr.DataArray(Bwaveletx+Bwavelety+Bwaveletz,coords=[Bwavelettimes,Bwavelet.frequency],\
                                dims=["time","frequency"])

    #---------------------------------------------------------------------------------------------------------------------
    # Compute plasma parameters
    #---------------------------------------------------------------------------------------------------------------------
    pparam = pyrf.plasma_calc(Bxyz,ne,ne,ne,ne)


    cmap = "jet"
    fig, axs = plt.subplots(6,sharex=True,figsize=(6.5,9))
    fig.subplots_adjust(bottom=0.05,top=0.95,left=0.15,right=0.85,hspace=0.)

    plot(axs, Bxyz, Exyzfaclf, Exyzfachf, specperpE, specparaE, specB, pparam)


    plt.show()

if __name__ == "__main__":

    main()
#---------------------------------------------------------------------------------------------------------------------
# End
#---------------------------------------------------------------------------------------------------------------------