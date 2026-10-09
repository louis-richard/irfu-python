#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import xarray as xr

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def feeps_split_integral_ch(inp_dataset):
    r"""This function splits the last integral channel from the FEEPS spectra,
    creating 2 new DataArrays

    Parameters
    ----------
    inp_dataset : xarray.Dataset
        Energetic particles energy spectrum from FEEPS.

    Returns
    -------
    out : xarray.Dataset
        Energetic particles energy spectra with the integral channel removed.
        The other variables (spin sectors, pitch angles, ...) are unchanged.
    out_500kev : xarray.Dataset
        Integral channel that was removed, with the spin sectors.

    Notes
    -----
    Only the eyes (``top-*`` and ``bottom-*`` variables) are split, as in IDL
    SPEDAS.

    """

    eyes = [k for k in inp_dataset if k.startswith(("top", "bottom"))]

    out_dict, out_dict_500kev = [{}, {}]

    if "spinsectnum" in inp_dataset:
        out_dict_500kev["spinsectnum"] = inp_dataset["spinsectnum"]

    for k in inp_dataset:
        if k in eyes:
            # Energy spectra with the integral channel removed
            out_dict[k] = inp_dataset[k][:, :-1]

            # Integral channel that was removed
            out_dict_500kev[k] = inp_dataset[k][:, -1]
        else:
            out_dict[k] = inp_dataset[k]

    out = xr.Dataset(out_dict, attrs=inp_dataset.attrs)

    out_500kev = xr.Dataset(out_dict_500kev, attrs=inp_dataset.attrs)

    return out, out_500kev
