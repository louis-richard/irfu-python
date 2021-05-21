#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 - 2021 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

"""get_eis_allt.py
@author: Louis Richard
"""

import xarray as xr

from .list_files import list_files
from .db_get_ts import db_get_ts


def get_eis_allt(tar_var, tint, mms_id, verbose: bool = True,
                 data_path: str = ""):
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
    verbose : bool, optional
        Set to True to follow the loading. Default is True.
    data_path : str
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

    pref = f"mms{mms_id:d}_epd_eis"

    var = {"mms_id": mms_id, "inst": "epd-eis", "dtype": data_type,
           "tmmode": data_rate, "lev": data_lvl, "specie": specie,
           "data_path": data_path}

    if data_rate == "brst":
        pref = f"{pref}_{data_rate}"

    pref = f"{pref}_{data_type}"

    # EIS includes the version of the files in the cdfname need to read it
    # before.
    files = list_files(tint, mms_id, var, data_path=data_path)

    file_version = int(files[0].split("_")[-1][1])
    var["version"] = file_version

    if data_unit.lower() in ["flux", "counts", "cps"]:
        suf = f"{specie}_P{file_version:d}_{data_unit.lower()}_t"
    else:
        raise ValueError("Invalid data unit")

    # Name of the data containing index of the probe, instrument, data rate,
    # data level and data type if needed
    dset_name = f"mms{var['mms_id']:d}_{var['inst']}_{var['tmmode']}" \
                f"_{var['lev']}_{var['dtype']}"

    # Names of the energy spectra in the CDF (one for each telescope)
    cdfnames = ["{}_{}{:d}".format(pref, suf, t) for t in range(6)]

    spin_nums = db_get_ts(dset_name, f"{pref}_spin", tint, data_path=data_path)

    outdict = {"spin": spin_nums}
    for i, cdfname in enumerate(cdfnames):
        scope_key = f"t{i:d}"

        outdict[scope_key] = db_get_ts(dset_name, cdfname, tint,
                                       data_path=data_path)
        outdict[scope_key] = outdict[scope_key].rename({"time": "time",
                                                        "Energy": "energy"})

    # Build Dataset
    out = xr.Dataset(outdict, attrs=var)

    return out
