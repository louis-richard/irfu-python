import numpy as np
from .get_feeps_active_eyes import get_feeps_active_eyes
from .db_get_ts import db_get_ts

def get_feeps_oneeye(tar_var="fluxe_brst_l2", eId="bottom-4", trange=None, mmsId=1):
    """
    Load energy spectrum all the target eye

    Parameters :
        tar_var : str
            target variable "{data_units}{specie}_{data_rate}_{level}"
                data_units : 
                    flux -> intensity (1/cm sr)
                    count -> counts (-)
                    CPS -> counts per second (1/s)
                specie : 
                    i -> ion
                    e -> electron
                data_rate : brst/srvy
                level : l1/l1b/l2/l3??

        eId : str
            index of the eye "{deck}-{id}"
                deck : top/bottom
                id : see get_feeps_active_eyes

        trange : list of str
            Time interval

        mmsId : int/str
            Index of the spacecraft
        
    """
    if trange is None:
        raise ValueError("empty time interval")
    
    if isinstance(mmsId,str): mmsId = int(mmsId)

    Var = {}
    Var["inst"] = "feeps"

    data_units = tar_var.split("_")[0][:-1]
    specie = tar_var.split("_")[0][-1]

    if specie == "e":
        Var["dtype"] = "electron"
    elif specie == "i":
        Var["dtype"] = "ion"
    else :
        raise ValueError("invalid specie")

    Var["tmmode"] = tar_var.split("_")[1]
    Var["lev"] = tar_var.split("_")[2]

    dsetName = "mms{:d}_feeps_{}_l2_{}".format(mmsId,Var["tmmode"],Var["dtype"])
    dsetPref = "mms{:d}_epd_feeps_{}_{}_{}".format(mmsId,Var["tmmode"],Var["lev"],Var["dtype"])

    active_eyes = get_feeps_active_eyes(Var,trange,mmsId)


    if eId.split("-")[0] in ["top","bottom"]:
        suf = eId.split("-")[0]
        eId = int(eId.split("-")[1])
        if eId in active_eyes[suf]:
            if data_units.lower() == "flux":
                suf = "_".join([suf,"intensity","sensorid",str(eId)])
            elif data_units.lower() == "counts":
                suf = "_".join([suf,"counts","sensorid",str(eId)])
            elif data_units.lower() == "cps":
                suf = "_".join([suf,"count_rate","sensorid",str(eId)])
            elif data_units == "mask":
                suf = "_".join([suf,"sector_mask","sensorid",str(eId)])
            else :
                raise ValueError("undefined variable")
        else :
            raise ValueError("Unactive eye")

    out = db_get_ts(dsetName,"_".join([dsetPref,suf]),trange)
    out.attrs["tmmode"] = Var["tmmode"]
    out.attrs["lev"] = Var["lev"]
    out.attrs["mmsId"] = mmsId
    out.attrs["dtype"] = Var["dtype"]

    
    return out