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

"""eis_omni.py
@author: Louis Richard
"""


def eis_omni(eis_allt):
    r"""Calculates the omni-directional flux for all 6 telescopes.

    Parameters
    ----------
    eis_allt : xarray.Dataset
        Dataset of the fluxes of all 6 telescopes.

    Returns
    -------
    flux_omni : xarray.DataArray
        Omni-directional flux for all 6 telescopes

    Examples
    --------
    >>> from pyrfu import mms

    Define spacecraft index and time interval

    >>> tint = ["2017-07-23T16:10:00", "2017-07-23T18:10:00"]
    >>> ic = 2

    Get EIS ExTOF all 6 telescopes fluxes

    >>> extof_allt = mms.get_eis_allt("flux_extof_proton_srvy_l2", tint, ic)

    Compute the omni-directional flux for all 6 telescopes

    >>> extof_omni = mms.eis_omni(extof_allt)

    """

    scopes = list(filter(lambda x: x[0] == "t", eis_allt))

    flux_omni = None

    for scope in scopes:
        try:
            flux_omni += eis_allt[scope].copy()
        except TypeError:
            flux_omni = eis_allt[scope].copy()

    flux_omni.data /= len(scopes)
    flux_omni.name = "flux_omni"

    return flux_omni
