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

from pyrfu.mms import psd_rebin
from pyrfu.pyrf import ts_skymap


def vdf_to_e64(vdf_e32):
    """Recompile data into 64 energy channels. Time resolution is halved.
    Only applies to skymap.

    Parameters
    ----------
    vdf_e32 : xarray.Dataset
        Time series of the particle distribution with 32 energy channels.

    Returns
    -------
    vdf_e64 : xarray.Dataset
        Time series of the particle distribution with 32 energy channels.

    """

    time_r, vdf_r, energy_r, phi_r = psd_rebin(vdf_e32, vdf_e32.phi,
                                               vdf_e32.attrs.get("energy0"),
                                               vdf_e32.attrs.get("energy1"),
                                               vdf_e32.attrs.get("esteptable"))

    energy_r = np.tile(energy_r, (len(vdf_r), 1))

    vdf_e64 = ts_skymap(time_r, vdf_r, energy_r, phi_r, vdf_e32.theta.data,
                        energy0=energy_r[0, :], energy1=energy_r[0, :],
                        esteptable=vdf_e32.attrs.get("esteptable"))

    # update delta_energy
    log_energy = np.log10(energy_r[0, :])
    log10_energy = np.diff(log_energy)
    log10_energy_plus = log_energy + 0.5 * np.hstack([log10_energy, log10_energy[-1]])
    log10_energy_minus = log_energy - 0.5 * np.hstack([log10_energy[0], log10_energy])

    energy_plus = 10 ** log10_energy_plus
    energy_minus = 10 ** log10_energy_minus
    delta_energy_plus = energy_plus - energy_r
    delta_energy_minus = abs(energy_minus - energy_r)
    delta_energy_plus[-1] = np.max(vdf_e32.attrs.get("delta_energy_minus")[:, -1])
    delta_energy_minus[0] = np.min(vdf_e32.attrs.get("delta_energy_minus")[:, 0])
    vdf_e64.attrs["delta_energy_plus"] = delta_energy_plus
    vdf_e64.attrs["delta_energy_minus"] = delta_energy_minus
    vdf_e64.attrs["esteptable"] = np.zeros(len(time_r))
    vdf_e64.attrs["species"] = vdf_e32.attrs["species"]
    vdf_e64.attrs["UNITS"] = vdf_e32.attrs["UNITS"]

    return vdf_e64
