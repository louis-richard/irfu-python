import warnings
import numpy as np
import xarray as xr
from astropy.time import Time

from .db_get_ts import db_get_ts


def feeps_spin_avg(inp_dset_omni=None):
    """
    This function will spin-average the omni-directional FEEPS energy spectra
    
    Parameters:
        inp_dset_omni : DataArray
            Spectrogram of all eyes in OMNI

    Returns:
        out : DataArray
            Spin-averaged OMNI energy spectrum
    """

    if inp_dset_omni is None:
        raise ValueError("feeps_spin_avg requires at least one argument")

    var = inp_dset_omni.attrs

    # get the spin sectors
    # v5.5+ = mms1_epd_feeps_srvy_l1b_electron_spinsectnum
    trange = list(Time(inp_dset_omni.time.data[[0, -1]], format="datetime64").isot)

    dset_name = "mms{:d}_feeps_{}_{}_{}".format(var["mmsId"], var["tmmode"], var["lev"], var["dtype"])
    dset_pref = "mms{:d}_epd_feeps_{}_{}_{}".format(var["mmsId"], var["tmmode"], var["lev"], var["dtype"])

    spin_sectors = db_get_ts(dset_name, "_".join([dset_pref, "spinsectnum"]), trange)

    spin_starts = [spin_end + 1 for spin_end in np.where(spin_sectors[:-1] >= spin_sectors[1:])[0]]

    energies = inp_dset_omni.coords["energy"]
    data = inp_dset_omni.data

    spin_avg_flux = np.zeros([len(spin_starts), len(energies)])

    current_start = spin_starts[0]

    for spin_idx in range(1, len(spin_starts) - 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            spin_avg_flux[spin_idx - 1, :] = np.nanmean(data[current_start:spin_starts[spin_idx] + 1, :], axis=0)
        
        current_start = spin_starts[spin_idx] + 1

    out = xr.DataArray(spin_avg_flux, coords=[inp_dset_omni.coords["time"][spin_starts], energies],
                       dims=inp_dset_omni.dims)

    return out
