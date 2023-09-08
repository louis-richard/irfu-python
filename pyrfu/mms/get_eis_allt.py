#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import xarray as xr

from .db_get_ts import db_get_ts
from .db_get_variable import db_get_variable

# Local imports
from .list_files import list_files

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def get_eis_allt(
    tar_var,
    tint,
    mms_id,
    verbose: bool = True,
    data_path: str = "",
):
    r"""Read energy spectrum of the selected specie in the selected energy
    range for all telescopes.

    Parameters
    ----------
    tar_var : str
        Key of the target variable like
        {data_unit}_{dtype}_{specie}_{data_rate}_{data_lvl}.
    tint : list of str
        Time interval.
    mms_id : int or float or str
        Index of the spacecraft.
    verbose : bool, Optional
        Set to True to follow the loading. Default is True.
    data_path : str, Optional
        Path of MMS data.

    Returns
    -------
    out : xarray.Dataset
        Dataset containing the energy spectrum of the 6 telescopes of the
        Energy Ion Spectrometer.

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint_brst = ["2017-07-23T16:54:24.000", "2017-07-23T17:00:00.000"]

    Read proton energy spectrum for all EIS telescopes

    >>> eis_allt = mms.get_eis_allt("Flux_extof_proton_srvy_l2", tint_brst, 2)

    """

    # Convert mms_id to integer
    mms_id = int(mms_id)

    data_unit, data_type, specie, data_rate, data_lvl = tar_var.split("_")

    if tint[0] < "2020-01-01":
        pref = f"mms{mms_id:d}_epd_eis"
    else:
        pref = f"mms{mms_id:d}_epd_eis_{data_rate}_{data_lvl}"

    var = {
        "mms_id": mms_id,
        "inst": "epd-eis",
        "dtype": data_type,
        "tmmode": data_rate,
        "lev": data_lvl,
        "specie": specie,
        "data_path": data_path,
    }

    # EIS includes the version of the files in the cdfname need to read it
    # before.
    files = list_files(tint, mms_id, var, data_path=data_path)

    if not files:
        raise FileNotFoundError("no files for these inputs!!")

    file_version = int(files[0].split("_")[-1][1])
    var["version"] = file_version

    pref = f"{pref}_{data_rate}_{data_lvl}_{data_type}"
    if int(files[0].split("_")[-1][3]) >= 1:
        if specie == "alpha":
            specie = "helium"
    else:
        if data_rate == "brst":
            pref = f"{pref}_{data_rate}_{data_type}"
        else:
            pref = f"{pref}_{data_type}"

    if data_unit.lower() in ["flux", "counts", "cps"]:
        suf = f"{specie}_P{file_version:d}_{data_unit.lower()}_t"
    else:
        raise ValueError("Invalid data unit")

    # Name of the data containing index of the probe, instrument, data rate,
    # data level and data type if needed
    dset_name = (
        f"mms{var['mms_id']:d}_{var['inst']}_{var['tmmode']}"
        f"_{var['lev']}_{var['dtype']}"
    )

    # Names of the energy spectra in the CDF (one for each telescope)
    cdfnames = [f"{pref}_{suf}{t:d}" for t in range(6)]

    spin_nums = db_get_ts(dset_name, f"{pref}_spin", tint, data_path=data_path)
    sectors = db_get_ts(dset_name, f"{pref}_sector", tint, data_path=data_path)

    e_minus = db_get_variable(
        dset_name,
        f"{pref}_{specie}_t0_energy_dminus",
        tint,
        verbose=verbose,
        data_path=data_path,
    )

    e_plus = db_get_variable(
        dset_name,
        f"{pref}_{specie}_t0_energy_dplus",
        tint,
        verbose=verbose,
        data_path=data_path,
    )

    outdict = {"spin": spin_nums, "sector": sectors}

    for i, cdfname in enumerate(cdfnames):
        scope_key = f"t{i:d}"

        outdict[scope_key] = db_get_ts(
            dset_name,
            cdfname,
            tint,
            verbose=verbose,
            data_path=data_path,
        )
        outdict[scope_key] = outdict[scope_key].rename(
            {"time": "time", "Energy": "energy"},
        )

        outdict[f"look_{scope_key}"] = db_get_ts(
            dset_name,
            f"{pref}_look_{scope_key}",
            tint,
            verbose=verbose,
            data_path=data_path,
        )

    e_plus = e_plus.assign_coords(x=outdict["t0"].energy.data)
    e_minus = e_minus.assign_coords(x=outdict["t0"].energy.data)
    e_plus = e_plus.rename({"x": "energy"})
    e_minus = e_minus.rename({"x": "energy"})

    # glob_attrs = {**outdict["spin"].attrs["GLOBAL"], **var}
    glob_attrs = {
        "delta_energy_plus": e_plus.data,
        "delta_energy_minus": e_minus.data,
        "species": specie,
        **outdict["spin"].attrs["GLOBAL"],
    }

    # Build Dataset
    out = xr.Dataset(outdict, attrs=glob_attrs)

    return out
