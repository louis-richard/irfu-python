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

g_corr = {"mms1-top6": 0.7, "mms1-top7": 2.5, "mms1-top8": 1.5,
          "mms1-bot5": 1.2, "mms1-bot6": 0.9, "mms1-bot7": 2.2,
          "mms1-bot8": 1.0, "mms2-top4": 1.2, "mms2-top6": 1.3, "mms2-top7": 0,
          "mms2-top8": 0.8, "mms2-bot6": 1.4, "mms2-bot7": 0, "mms2-bot8": 1.5,
          "mms3-top6": 0.7, "mms3-top7": 0.8, "mms3-top8": 1.0,
          "mms3-bot6": 0.9, "mms3-bot7": 0.9, "mms3-bot8": 1.3,
          "mms4-top6": 0.8, "mms4-top7": 0, "mms4-top8": 1.0, "mms4-bot6": 0.8,
          "mms4-bot7": 0.6, "mms4-bot8": 0.9, "mms4-bot9": 1.5}


def feeps_flat_field_corrections(inp_alle):
    r"""Apply flat field correction factors to FEEPS ion/electron
    data. Correct factors are from the gain factor found in:
    FlatFieldResults_V3.xlsx from Drew Turner, 1/19/2017

    Parameters
    ----------
    inp_alle : xarray.Dataset
        Dataset containing the energy spectrum of the available eyes of the
        Fly's Eye Energetic Particle Spectrometer (FEEPS).

    Returns
    -------
    out : xarray.Dataset
        Dataset containing the energy spectrum of the available eyes of the
        Fly's Eye Energetic Particle Spectrometer (FEEPS) with corrected
        data.

    """

    # Spacecraft index
    mms_id = inp_alle.attrs["mmsId"]

    # List of sensors and eyes
    sensors_eyes = list(filter(lambda x: x[:3] in ["top", "bot"], inp_alle))

    out_dict = inp_alle.copy()

    for se in sensors_eyes:
        sensor, eye = se.split("-")
        correction = g_corr.get(f"mms{mms_id}-{sensor[:3]}{int(eye)}", 1.0)

        out_dict[se].data *= correction

    out = xr.Dataset(out_dict)
    out.attrs = inp_alle.attrs

    return out
