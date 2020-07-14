import warnings
import numpy as np
import xarray as xr

from .get_feeps_active_eyes import get_feeps_active_eyes
from .get_feeps_eye import get_feeps_eye

def get_feeps_omni(tar_var="flux_ion_brst_l2",mmsId=1,trange=None):
    Var = {}
    data_units, Var["dtype"], Var["tmmode"], Var["lev"] = tar_var.split("_")
    specie = Var["dtype"][0]

    if Var["dtype"] == 'electron':
        energies = np.array([33.2, 51.90, 70.6, 89.4, 107.1, 125.2, 146.5, 171.3,
                    200.2, 234.0, 273.4, 319.4, 373.2, 436.0, 509.2, 575.8])
    else:
        energies = np.array([57.9, 76.8, 95.4, 114.1, 133.0, 153.7, 177.6,
                    205.1, 236.7, 273.2, 315.4, 363.8, 419.7, 484.2,  558.6, 609.9])

    # set unique energy bins per spacecraft; from DLT on 31 Jan 2017
    eEcorr = [14.0, -1.0, -3.0, -3.0]
    iEcorr = [0.0, 0.0, 0.0, 0.0]
    eGfact = [1.0, 1.0, 1.0, 1.0]
    iGfact = [0.84, 1.0, 1.0, 1.0]

    energies += eval("{}Ecorr[{:d}]".format(specie,mmsId))


    active_eyes = get_feeps_active_eyes(Var,trange,mmsId)

    # percent error around energy bin center to accept data for averaging; 
    # anything outside of energies[i] +/- en_chk*energies[i] will be changed 
    # to NAN and not averaged   
    en_chk = 0.1

    top_sensors = active_eyes["top"]
    bot_sensors = active_eyes["bottom"]


    for tsen in top_sensors:
        top = get_feeps_eye("{}{}_{}_{}".format(data_units,Var["dtype"][0],Var["tmmode"],Var["lev"]),\
                            mmsId=mmsId, eId="top-{:d}".format(tsen),trange=trange)
        mask = get_feeps_eye("mask{}_{}_{}".format(Var["dtype"][0],Var["tmmode"],Var["lev"]),\
                            mmsId=mmsId, eId="top-{:d}".format(tsen),trange=trange)
        #top.data[mask.data==1] = np.nan
        mask.data = np.tile(mask.data[:,0],(mask.shape[1],1)).T
        top.data[mask.data==1] = np.nan
        exec("Tit{:d} = top".format(tsen))


    for bsen in bot_sensors:
        bot = get_feeps_eye("{}{}_{}_{}".format(data_units,Var["dtype"][0],Var["tmmode"],Var["lev"]),\
                            mmsId=mmsId, eId="bottom-{:d}".format(tsen),trange=trange)
        mask = get_feeps_eye("mask{}_{}_{}".format(Var["dtype"][0],Var["tmmode"],Var["lev"]),\
                            mmsId=mmsId, eId="bottom-{:d}".format(tsen),trange=trange)
        #bot.data[mask.data==1] = np.nan
        mask.data = np.tile(mask.data[:,0],(mask.shape[1],1)).T
        bot.data[mask.data==1] = np.nan
        exec("Bit{:d} = top".format(bsen))

    dalleyes = np.empty((top.shape[0],top.shape[1],len(top_sensors)+len(bot_sensors)))

    for i, tsen in enumerate(top_sensors):
        dalleyes[...,i] = eval("Tit{:d}".format(tsen))

    for i, bsen in enumerate(bot_sensors):
        dalleyes[...,len(top_sensors)+i] = eval("Bit{:d}".format(bsen))



    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        flux_omni = np.nanmean(dalleyes, axis=2)

    flux_omni *= eval("{}Gfact[{:d}]".format(specie,mmsId))

    time = eval("Tit{:d}.time".format(top_sensors[0]))

    out = xr.DataArray(flux_omni[:],coords=[time,energies],dims=["time","energy"])

    return out