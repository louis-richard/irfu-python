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

"""eis_spec_combine_sc.py
@author: Louis Richard
"""

import itertools
import numpy as np
import xarray as xr


def eis_spec_combine_sc(omni_vars):
    r"""Combines omni-directional energy spectrogram variable from EIS on
    multiple MMS spacecraft.


    Parameters
    ----------
    omni_vars : list
        Omni-directional energy spectrograms of all spacecraft

    Returns
    -------
    omni_spec : xarray.DataArray
        Combined omni-directional energy spectrogram.

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint = ["2017-07-23T16:10:00", "2017-07-23T18:10:00"]

    Load EIS ExTOF flux spectrograms for all 6 telescopes for all spacecraft

    >>> extof_allt_mms = [mms.get_eis_allt("flux_extof_proton_srvy_l2", tint, ic) for ic in range(2, 5)]

    Compute the omni-direction flux spectrogram for all spacecraft

    >>> extof_omni_mms = [mms.eis_omni(extof_allt) for extof_allt in extof_allt_mms]

    Combine spectrograms of all spacecraft

    >>> extof_omni_mmsx = mms.eis_spec_combine_sc(extof_omni_mms)

    """

    reftime_sc_loc = np.argmin([len(x_.time.data) for x_ in omni_vars])
    refener_sc_loc = np.argmin([len(x_.energy.data) for x_ in omni_vars])

    time_refprobe = omni_vars[reftime_sc_loc]
    ener_refprobe = omni_vars[refener_sc_loc]

    nt_ref, ne_ref = [time_refprobe.shape[0], ener_refprobe.shape[1]]

    # time x energy x spacecraft
    omni_spec_data = np.empty([nt_ref, ne_ref, len(omni_vars)])
    omni_spec_data[:] = np.nan
    # time x energy
    omni_spec = np.empty([nt_ref, ne_ref])
    omni_spec[:] = np.nan

    energy_data = np.zeros([ne_ref, len(omni_vars)])
    common_energy = np.zeros(ne_ref)

    # Average omni flux over all spacecraft and define common energy grid
    for pp, flux_ in enumerate(omni_vars):
        energy_data[:, pp] = flux_.Energy.data[0:ne_ref]
        omni_spec_data[0:nt_ref, :, pp] = flux_.data[0:nt_ref, 0:ne_ref]

    for ee in range(len(common_energy)):
        common_energy[ee] = np.nanmean(energy_data[ee, :], axis=0)

    # Average omni flux over all spacecraft
    for tt, ee in itertools.product(range(nt_ref), range(ne_ref)):
        omni_spec[tt, ee] = np.nanmean(omni_spec_data[tt, ee, :], axis=0)

    omni_spec = xr.DataArray(omni_spec,
                             coords=[time_refprobe.time.data,
                                     ener_refprobe.energy.data],
                             dims=["time", "energy"])

    return omni_spec
