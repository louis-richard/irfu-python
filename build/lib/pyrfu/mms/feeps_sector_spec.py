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


def feeps_sector_spec(inp_alle):
    r"""Creates sector-spectrograms with FEEPS data (particle data organized
    by time and sector number)

    Parameters
    ----------
    inp_alle : xarray.Dataset
        Dataset of energy spectrum of all eyes.

    Returns
    -------
    out : xarray.Dataset
        Sector-spectrograms with FEEPS data for all eyes.

    """

    sensors_eyes_top = list(filter(lambda x: x[:3] in "top", inp_alle))
    sensors_eyes_bot = list(filter(lambda x: x[:3] in "bot", inp_alle))
    sensors_eyes = [*sensors_eyes_top, *sensors_eyes_bot]

    sector_time = inp_alle["spinsectnum"].time.data
    sector_data = inp_alle["spinsectnum"].data

    out_dict = {k: inp_alle[k] for k in inp_alle if k not in sensors_eyes}

    for se in sensors_eyes:
        sensor_data = inp_alle[se].data

        spin_starts = np.where(sector_data[:-1] > sector_data[1:])[0] + 1

        sector_spec = np.zeros((len(spin_starts), 64))

        c_start = spin_starts[0]

        for i, spin in enumerate(spin_starts):
            # find the sectors for this spin
            s_ = sector_data[c_start:spin]

            sector_spec[i, s_] = np.nanmean(sensor_data[c_start:spin, :],
                                            axis=1)

            c_start = spin

        out_dict[se] = xr.DataArray(sector_spec,
                                    coords=[sector_time[spin_starts],
                                            np.arange(64)],
                                    dims=["time", "sectornum"])

    out = xr.Dataset(out_dict)

    return out
