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

"""calc_hpca_anodes.py
@author: Louis Richard
"""


def _hpca_anodes(fov=None):
    if fov is None:
        fov = [0, 360]

    anodes = [123.75000, 101.25000, 78.750000, 56.250000, 33.750000,
              11.250000, 11.250000, 33.750000, 56.250000, 78.750000,
              101.25000, 123.75000, 146.25000, 168.75000, 168.75000,
              146.25000]

    anodes[6:14] = [anode + 180. for anode in anodes[6:14]]

    out = []
    for i, anode in enumerate(anodes):
        if float(fov[0]) <= anode <= float(fov[1]):
            out.append(i)

    return out


def calc_hpca_anodes(vdf, method: str = "mean", fov: list = None):
    r"""
    Sum or average the Hot Plasma Composition Analyses (HPCA) data over the
    requested field-of-view (fov).

    Parameters
    ----------
    vdf : xarray.DataArray
        Ion PSD or flux; [nt, npo16, ner63], looking direction.

    method : str, optional
        Method "sum" or "mean". Use "sum" for Phase Space Density and "mean"
        for Differential Particle Flux. Default is "mean"

    fov : list
        Field of view, in angles, from 0-360

    Returns
    -------
    new_vdf : xarray.DataArray
        Ion PSD or flux over the field-of-view.

    """

    if fov is None:
        fov = [0, 360]

    anodes_in_fov = _hpca_anodes(fov=fov)

    if method == "mean":
        new_vdf = vdf[:, anodes_in_fov, :].mean(axis=1)
    elif method == "sum":
        new_vdf = vdf[:, anodes_in_fov, :].sum(axis=1)
    else:
        raise TypeError("Invalid method")

    return new_vdf
