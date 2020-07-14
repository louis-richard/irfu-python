import warnings
import numpy as np
from astropy.time import Time

from .db_get_ts import db_get_ts



def feeps_spin_avg(inp_dset_omni):
    """
    This function will spin-average the omni-directional FEEPS energy spectra
    
    Parameters:
        probe: str
            probe #, e.g., '4' for MMS4
        data_units: str
            'intensity' or 'count_rate'
        datatype: str
            'electron' or 'ion'
        data_rate: str
            instrument data rate, e.g., 'srvy' or 'brst'
        level: str
            data level, e.g., 'l2'
            
        suffix: str
            suffix of the loaded data
    Returns:
        Name of tplot variable created.
    """

    Var = inp_dset_omni.attrs

    if Var["dtype"] == "electron":
        lower_en = 71.0
    else:
        lower_en = 78.0
    
    # get the spin sectors
    # v5.5+ = mms1_epd_feeps_srvy_l1b_electron_spinsectnum
    trange  = list(Time(inp_dset_omni.time.data[[0,-1]],format="datetime64").isot)  

    dsetName = "mms{:d}_feeps_{}_{}_{}".format(Var["mmsId"],Var["tmmode"],Var["lev"],Var["dtype"])
    dsetPref = "mms{:d}_epd_feeps_{}_{}_{}".format(Var["mmsId"],Var["tmmode"],Var["lev"],Var["dtype"])  

    spin_sectors = db_get_ts(dsetName,"_".join([dsetPref,"spinsectnum"]),trange)


    spin_starts = [spin_end + 1 for spin_end in np.where(spin_sectors[:-1] >= spin_sectors[1:])[0]]

    


    var_name = prefix + data_rate + '_' + level + '_' + datatype + '_' + data_units + '_omni'

    times, data, energies = get_data(var_name + suffix)

    spin_avg_flux = np.zeros([len(spin_starts), len(energies)])

    current_start = spin_starts[0]
    for spin_idx in range(1, len(spin_starts)-1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            spin_avg_flux[spin_idx-1, :] = np.nanmean(data[current_start:spin_starts[spin_idx]+1, :], axis=0)
        current_start = spin_starts[spin_idx] + 1