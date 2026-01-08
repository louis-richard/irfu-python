#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
from copy import deepcopy

# Local imports
from .feeps_correct_energies import feeps_correct_energies
from .feeps_flat_field_corrections import feeps_flat_field_corrections
from .feeps_remove_bad_data import feeps_remove_bad_data
from .feeps_remove_sun import feeps_remove_sun
from .feeps_split_integral_ch import feeps_split_integral_ch

__author__ = "Apostolos Kolokotronis"
__email__ = "apostolosk@irf.se"
__copyright__ = "Copyright 2020-2025"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"


def feeps_corrections(feeps_alle):
    r"""
    Apply the FEEPS corrections to the all FEEPS eyes data product. The function
    applies the following corrections: feeps_correct_energies,
    feeps_flat_field_corrections, feeps_remove_bad_data and
    feeps_remove_sun.

    Parameters
    ----------
    feeps_alle : xarray.Dataset
        The all FEEPS eyes data product.

    Returns
    -------
    feeps_alle_clean : xarray.Dataset
        The all FEEPS eyes data product with the corrections applied.

    """

    # Make a deep copy of the input dataset to avoid modifying the original data
    feeps_alle_clean = deepcopy(feeps_alle)

    feeps_alle_clean = feeps_correct_energies(feeps_alle_clean)
    feeps_alle_clean = feeps_flat_field_corrections(feeps_alle_clean)
    feeps_alle_clean = feeps_remove_bad_data(feeps_alle_clean)

    split_int_ch = feeps_split_integral_ch(feeps_alle_clean)
    feeps_alle_clean = split_int_ch[0]
    # feeps_alle_500kev = split_int_ch[1];

    feeps_alle_clean = feeps_remove_sun(feeps_alle_clean)

    return feeps_alle_clean
