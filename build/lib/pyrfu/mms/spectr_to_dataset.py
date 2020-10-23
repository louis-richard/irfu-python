#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spectr_to_dataset.py

@author : Louis RICHARD
"""

import numpy as np
import xarray as xr


def spectr_to_dataset(spectr=None):
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

    assert spectr is not None and isinstance(spectr, xr.DataArray)

    time = spectr.time.data
    energy = spectr.energy.data
    energy = np.tile(energy, (len(time), 1))

    data = spectr.data

    out_dict = {"data": (["time", "idx0"], data), "energy": (["time", "idx0"], energy),
                "time": time, "idx0": np.arange(energy.shape[1])}

    out = xr.Dataset(out_dict)

    out.attrs = spectr.attrs

    return out
