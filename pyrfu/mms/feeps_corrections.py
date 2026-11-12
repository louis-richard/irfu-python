#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contributed by Apostolos Kolokotronis."""

# Local imports
from .feeps_correct_energies import feeps_correct_energies
from .feeps_flat_field_corrections import feeps_flat_field_corrections
from .feeps_remove_bad_data import feeps_remove_bad_data
from .feeps_remove_sun import feeps_remove_sun
from .feeps_split_integral_ch import feeps_split_integral_ch


def feeps_corrections(feeps_alle):
    r"""
    Apply the FEEPS corrections to the all FEEPS eyes data product. The function
    applies the following corrections: feeps_correct_energies,
    feeps_flat_field_corrections, feeps_remove_bad_data, then splits the
    integral channel (feeps_split_integral_ch) and applies feeps_remove_sun, in
    the order of IDL SPEDAS and pyspedas (``mms_load_feeps``). The integral
    channel is not returned.

    Parameters
    ----------
    feeps_alle : xarray.Dataset
        The all FEEPS eyes data product.

    Returns
    -------
    feeps_alle_clean : xarray.Dataset
        The all FEEPS eyes data product with the corrections applied.

    """

    # Each step returns a new Dataset, without changing its input (a deep copy
    # of the input was made first)
    feeps_alle_clean = feeps_correct_energies(feeps_alle)
    feeps_alle_clean = feeps_flat_field_corrections(feeps_alle_clean)
    feeps_alle_clean = feeps_remove_bad_data(feeps_alle_clean)

    split_int_ch = feeps_split_integral_ch(feeps_alle_clean)
    feeps_alle_clean = split_int_ch[0]

    feeps_alle_clean = feeps_remove_sun(feeps_alle_clean)

    return feeps_alle_clean
