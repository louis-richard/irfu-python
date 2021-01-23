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

import numpy as np
import xarray as xr


def spectr_to_dataset(spectr):
    """Convert energy spectrum (DataArray) to Dataset.

    Parameters
    ----------
    spectr : xarray.DataArray
        Spectrogram in DataArray format.

    Returns
    -------
    out : xarray.Dataset
        Spectrogram in Dataset format.

    """

    time = spectr.time.data
    energy = spectr.energy.data
    energy = np.tile(energy, (len(time), 1))

    data = spectr.data

    out_dict = {"data": (["time", "idx0"], data), "energy": (["time", "idx0"], energy),
                "time": time, "idx0": np.arange(energy.shape[1])}

    out = xr.Dataset(out_dict)

    out.attrs = spectr.attrs

    return out
