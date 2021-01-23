#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

import xarray as xr

from .list_files import list_files
from .db_get_ts import db_get_ts


def get_eis_allt(tar_var, tint, mms_id, verbose=True):
    """Read energy spectrum of the selected specie in the selected energy range for all telescopes.

    Parameters
    ----------
    tar_var : str
        Key of the target variable like {data_unit}_{dtype}_{specie}_{data_rate}_{data_lvl}.

    tint : list of str
        Time interval.

    mms_id : int or float or str
        Index of the spacecraft.

    verbose : bool
        Set to True to follow the loading. Default is True.

    Returns
    -------
    out : xarray.Dataset
        Dataset containing the energy spectrum of the 6 telescopes of the Energy Ion Spectrometer.

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

    var = {"mms_id": mms_id, "inst": "epd-eis", "tmmode": data_rate, "lev": data_lvl}

    if data_type == "electronenergy":
        if specie == "electron":
            var["dtype"], var["specie"] = [data_type, specie]

            pref = f"mms{mms_id:d}_epd_eis_{data_rate}_{data_type}_{specie}"
        else:
            raise ValueError("invalid specie")

    elif data_type == "extof":
        if specie == "proton":
            var["dtype"], var["specie"] = [data_type, specie]

            pref = f"mms{mms_id:d}_epd_eis_{data_rate}_{data_type}_{specie}"
        elif specie == "oxygen":
            var["dtype"], var["specie"] = [data_type, specie]

            pref = f"mms{mms_id:d}_epd_eis_{data_rate}_{data_type}_{specie}"
        elif specie == "alpha":
            var["dtype"], var["specie"] = [data_type, specie]

            pref = f"mms{mms_id:d}_epd_eis_{data_rate}_{data_type}_{specie}"
        else:
            raise ValueError("invalid specie")

    elif data_type == "phxtof":
        if specie == "proton":
            var["dtype"], var["specie"] = [data_type, specie]

            pref = f"mms{mms_id:d}_epd_eis_{data_rate}_{data_type}_{specie}"
        elif specie == "oxygen":
            var["dtype"], var["specie"] = [data_type, specie]

            pref = f"mms{mms_id:d}_epd_eis_{data_rate}_{data_type}_{specie}"
        else:
            raise ValueError("Invalid specie")
    else:
        raise ValueError("Invalid data type")

    # EIS includes the version of the files in the cdfname need to read it before.
    files = list_files(tint, mms_id, var)

    file_version = int(files[0].split("_")[-1][1])
    var["version"] = file_version

    if data_unit.lower() in ["flux", "counts", "cps"]:
        suf = "P{:d}_{}_t".format(file_version, data_unit.lower())
    else:
        raise ValueError("Invalid data unit")

    # Name of the data containing index of the probe, instrument, data rate, data level and data
    # type if needed
    dset_name = f"mms{var['mms_id']:d}_{var['inst']}_{var['tmmode']}_{var['lev']}_{var['dtype']}"

    # Names of the energy spectra in the CDF (one for each telescope)
    cdfnames = ["{}_{}{:d}".format(pref, suf, t) for t in range(6)]

    outdict = {}
    for i, cdfname in enumerate(cdfnames):
        scope_key = f"t{i:d}"

        if verbose:
            print(f"Loading {cdfname}...")

        outdict[scope_key] = db_get_ts(dset_name, cdfname, tint)

    # Build Dataset
    out = xr.Dataset(outdict, attrs=var)

    return out
