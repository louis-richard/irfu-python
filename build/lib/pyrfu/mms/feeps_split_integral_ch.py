#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feeps_split_integral_ch.py

@author : Louis RICHARD
"""

import xarray as xr


def feeps_split_integral_ch(inp_dataset=None):
    """This function splits the last integral channel from the FEEPS spectra, creating 2 new DataArrays

    Parameters
    ----------
    inp_dataset : xarray.DataArray
        Energetic particles energy spectrum from FEEPS.

    Returns
    -------
    out : xarray.DataArray
        Energetic particles energy spectra with the integral channel removed.

    out_500kev : xarray.DataArray
        Integral channel that was removed.

    """

    assert inp_dataset is not None and isinstance(inp_dataset, xr.Dataset)

    outdict, outdict_500kev = [{}, {}]

    for k in inp_dataset:
        try:
            # Energy spectra with the integral channel removed
            outdict[k] = inp_dataset[k][:, :-1]

            # Integral channel that was removed
            outdict_500kev[k] = inp_dataset[k][:, -1]
        except IndexError:
            pass

    out = xr.Dataset(outdict, attrs=inp_dataset.attrs)

    out_500kev = xr.Dataset(outdict_500kev, attrs=inp_dataset.attrs)

    return out, out_500kev
