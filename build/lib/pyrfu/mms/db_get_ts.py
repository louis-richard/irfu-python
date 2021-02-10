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

from .list_files import list_files
from .get_ts import get_ts

from ..pyrf import ts_append


def db_get_ts(dataset_name, cdf_name, tint=None, data_path=""):
    """Get variable time series in the cdf file.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    cdf_name : str
        Name of the target field in cdf file.

    tint : list of str
        Time interval.

    data_path : str, optional
        Path of MMS data. Default uses `pyrfu.mms.mms_config.py`

    Returns
    -------
    out : xarray.DataArray
        Time series of the target variable.

    """

    dataset = dataset_name.split("_")

    # Index of the MMS spacecraft
    probe = dataset[0][-1]

    var = {"inst": dataset[1], "tmmode": dataset[2], "lev": dataset[3]}

    try:
        var["dtype"] = dataset[4]
    except IndexError:
        pass

    files = list_files(tint, probe, var, data_path=data_path)

    out = None
    for i, file in enumerate(files):
        out = ts_append(out, get_ts(file, cdf_name, tint))

    return out
