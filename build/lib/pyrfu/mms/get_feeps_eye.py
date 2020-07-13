import numpy as np
from .get_feeps_active_eyes import get_feeps_active_eyes
from .db_get_ts import db_get_ts

def get_feeps_eye(tar_var="fluxe_brst_l2", mmsId="1", eId="bottom-4",trange=None):
    if trange is None:
        raise ValueError("empty time interval")
    
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

    dsetName = "mms{}_feeps_{}_l2_{}".format(mmsId,Var["tmmode"],Var["dtype"])
    dsetPref = "mms{}_epd_feeps_{}_{}_{}".format(mmsId,Var["tmmode"],Var["lev"],Var["dtype"])

    active_eyes = get_feeps_active_eyes(trange,mmsId,Var)


    if eId.split("-")[0] in ["top","bottom"]:
        suf = eId.split("-")[0]
        eId = int(eId.split("-")[1])
        if eId in active_eyes[suf]:
            if data_units == "flux":
                suf = "_".join([suf,"intensity","sensorid",str(eId)])
            elif data_units == "counts":
                suf = "_".join([suf,"count_rate","sensorid",str(eId)])
            elif data_units == "mask":
                suf = "_".join([suf,"sector_mask","sensorid",str(eId)])
            else :
                raise ValueError("undefined variable")
        else :
            raise ValueError("Unactive eye")

    out = db_get_ts(dsetName,"_".join([dsetPref,suf]),trange)
    
    return out