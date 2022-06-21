#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def feeps_split_integral_ch(inp_dataset):
    r"""This function splits the last integral channel from the FEEPS spectra,
    creating 2 new DataArrays

    Parameters
    ----------
    inp_dataset : xarray.Dataset
        Energetic particles energy spectrum from FEEPS.

    Returns
    -------
    out : xarray.Dataset
        Energetic particles energy spectra with the integral channel removed.
    out_500kev : xarray.Dataset
        Integral channel that was removed.

    """

    out_dict, out_dict_500kev = [{}, {}]

    out_dict["spinsectnum"] = inp_dataset["spinsectnum"]

    for k in inp_dataset:
        try:
            # Energy spectra with the integral channel removed
            out_dict[k] = inp_dataset[k][:, :-1]

            # Integral channel that was removed
            out_dict_500kev[k] = inp_dataset[k][:, -1]
        except IndexError:
            pass

    out = xr.Dataset(out_dict, attrs=inp_dataset.attrs)

    out_500kev = xr.Dataset(out_dict_500kev, attrs=inp_dataset.attrs)

    return out, out_500kev
