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


def feeps_split_integral_ch(inp_dataset):
    """This function splits the last integral channel from the FEEPS spectra, creating 2 new
    DataArrays

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
