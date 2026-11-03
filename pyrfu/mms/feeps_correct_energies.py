#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import xarray as xr

# Local imports
from .feeps_energy_table import feeps_energy_table

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def feeps_correct_energies(feeps_alle):
    r"""Corrects the energy tables of the FEEPS spectra.

    Modifies the energy table in FEEPS spectra (intensity, count_rate,
    counts) using :func:`pyrfu.mms.feeps_energy_table` (which is s/c, sensor
    head and sensor ID dependent).

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
        energy table. The energy dimension of each eye is named as in
        :func:`pyrfu.mms.get_feeps_alleyes` (e.g. ``energy_top-1``).

    Raises
    ------
    ValueError
        If an eye doesn't have the 16 energy channels of the table, e.g. after
        :func:`pyrfu.mms.feeps_split_integral_ch`: correct the energies first.

    """

    mms_id = feeps_alle.attrs["mmsId"]

    sensors_eyes = list(filter(lambda x: x[:3] in ["top", "bot"], feeps_alle))

    out_dict = {k: feeps_alle[k] for k in feeps_alle if k not in sensors_eyes}

    for sensors_eye in sensors_eyes:
        sensor, eye = sensors_eye.split("-")

        new_energy = feeps_energy_table(mms_id, sensor, int(eye))

        dim = feeps_alle[sensors_eye].dims[1]

        if feeps_alle.sizes[dim] != len(new_energy):
            raise ValueError(
                f"{sensors_eye} has {feeps_alle.sizes[dim]:d} energy channels "
                f"instead of {len(new_energy):d}: correct the energies before "
                "feeps_split_integral_ch"
            )

        # each eye has its own energy table, so its own dimension, named as in
        # get_feeps_alleyes
        out_dict[sensors_eye] = (
            feeps_alle[sensors_eye]
            .assign_coords({dim: new_energy})
            .rename({dim: f"energy_{sensors_eye}"})
        )

        # The attributes of the energies (UNITS keV, lost)
        out_dict[sensors_eye][f"energy_{sensors_eye}"].attrs = dict(
            feeps_alle[sensors_eye][dim].attrs
        )

    out = xr.Dataset(out_dict)
    out.attrs = dict(feeps_alle.attrs)

    return out
