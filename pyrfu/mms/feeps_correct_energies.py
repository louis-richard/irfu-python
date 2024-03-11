#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import xarray as xr

# Local imports
from .feeps_energy_table import feeps_energy_table

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def feeps_correct_energies(feeps_alle):
    r"""Modifies the energy table in FEEPS spectra (intensity, count_rate,
    counts) using the function: mms_feeps_energy_table (which is s/c, sensor
    head and sensor ID dependent)

    Parameters
    ----------
    feeps_alle : xarray.Dataset
        Dataset containing the energy spectrum of the available eyes of the
        Fly's Eye Energetic Particle Spectrometer (FEEPS).

    Returns
    -------
    out : xarray.Dataset
        Dataset containing the energy spectrum of the available eyes of the
        Fly's Eye Energetic Particle Spectrometer (FEEPS) with corrected
        energy table.

    """

    mms_id = feeps_alle.attrs["mmsId"]

    sensors_eyes = list(filter(lambda x: x[:3] in ["top", "bot"], feeps_alle))

    out_dict = {k: feeps_alle[k] for k in feeps_alle if k not in sensors_eyes}

    for sensors_eye in sensors_eyes:
        sensor, eye = sensors_eye.split("-")

        new_energy = feeps_energy_table(mms_id, sensor, int(eye))

        dim = feeps_alle[sensors_eye].dims[1]
        out_dict[sensors_eye] = feeps_alle[sensors_eye].assign_coords(
            {dim: new_energy},
        )
        out_dict[sensors_eye] = out_dict[sensors_eye].rename(
            {dim: f"energy-{sensors_eye}"},
        )

    out = xr.Dataset(out_dict)
    out.attrs = feeps_alle.attrs

    return out
