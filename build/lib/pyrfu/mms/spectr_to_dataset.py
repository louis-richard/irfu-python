#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import numpy as np
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def spectr_to_dataset(spectr):
    r"""Convert energy spectrum (DataArray) to Dataset.

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

    out_dict = {"data": (["time", "idx0"], data),
                "energy": (["time", "idx0"], energy),
                "time": time, "idx0": np.arange(energy.shape[1])}

    out = xr.Dataset(out_dict)

    out.attrs = spectr.attrs

    return out
