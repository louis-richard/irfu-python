#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .feeps_spin_avg import _complete_spins

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def feeps_sector_spec(inp_alle):
    r"""Creates sector-spectrograms with FEEPS data.

    The particle data are organized by time and sector number.

    Parameters
    ----------
    inp_alle : xarray.Dataset
        Dataset of energy spectrum of all eyes, with the spin sectors
        (``spinsectnum``).

    Returns
    -------
    out : xarray.Dataset
        Sector-spectrograms with FEEPS data for all eyes: one row per complete
        spin and one column per spin sector (64).

    Raises
    ------
    ValueError
        If the integral channel of the eyes is not split (16 channels), or if
        the interval has no complete spin.

    Notes
    -----
    A spin starts at the first sample after the spin sector number wraps
    around and is time stamped at that sample. Each sample of a spin, averaged
    over energy, goes to the column of its spin sector; the sectors not
    sampled are NaN. Only the complete spins are kept, as in
    :func:`pyrfu.mms.feeps_spin_avg`: the partial spins at the start and end
    of the interval and the spins cut by data gaps are left out (they were
    kept).

    IDL SPEDAS leaves the first row at zero, stamps each spin at the start of
    the next one, and lets the first sample of the next spin overwrite its
    sector.

    """

    sensors_eyes = list(filter(lambda x: x.startswith(("top", "bottom")), inp_alle))

    for sensors_eye in sensors_eyes:
        if inp_alle[sensors_eye].shape[1] == 16:
            raise ValueError(
                f"{sensors_eye} has 16 energy channels: split the integral channel "
                "first with feeps_split_integral_ch (or use feeps_corrections)"
            )

    sector_time = inp_alle["spinsectnum"].time.data
    sector_data = inp_alle["spinsectnum"].data.astype(int)
    spin_starts, spin_stops = _complete_spins(sector_data)

    out_dict = {}

    for sensors_eye in sensors_eyes:
        # average over energy, quiet for the samples without data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sensor_data = np.nanmean(inp_alle[sensors_eye].data, axis=1)

        sector_spec = np.full((len(spin_starts), 64), np.nan)

        for i, (i_start, i_stop) in enumerate(zip(spin_starts, spin_stops)):
            sector_spec[i, sector_data[i_start:i_stop]] = sensor_data[i_start:i_stop]

        out_dict[sensors_eye] = xr.DataArray(
            sector_spec,
            coords=[sector_time[spin_starts], np.arange(64)],
            dims=["time", "sectornum"],
        )

    out = xr.Dataset(out_dict, attrs={**inp_alle.attrs})

    return out
