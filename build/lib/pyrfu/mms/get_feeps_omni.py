#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Local imports
from .get_feeps_alleyes import get_feeps_alleyes
from .feeps_remove_bad_data import feeps_remove_bad_data
from .feeps_split_integral_ch import feeps_split_integral_ch
from .feeps_remove_sun import feeps_remove_sun
from .feeps_omni import feeps_omni
from .feeps_spin_avg import feeps_spin_avg

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def get_feeps_omni(tar_var, tint, mms_id, verbose: bool = True,
                   data_path: str = "", spin_avg: bool = False):
    r"""Computes the omni-directional energy spectrum of the target data unit
    for the target specie over the target energy range. The data are washed,
    splitted and sunlight contamination free.

    Parameters
    ----------
    tar_var : str
        Key of the target variable like
        {data_unit}_{specie}_{data_rate}_{data_lvl}.
    tint : list of str
        Time interval.
    mms_id : int or str
        Index of the spacecraft.
    verbose : bool, Optional
        Set to True to follow the loading. Default is True.
    data_path : str, Optional
        Path of MMS data. Default uses `pyrfu.mms.mms_config.py`
    spin_avg : bool, Optional
        Spin average the omni-directional flux. Default is False.

    Returns
    --------
    flux_omni : xarray.DataArray
        Energy spectrum of the target data unit for the target specie in omni
        direction.

    See Also
    --------
    pyrfu.mms.get_feeps_alleyes, pyrfu.mms.feeps_remove_bad_data,
    pyrfu.mms.feeps_split_integral_ch, pyrfu.mms.feeps_remove_sun,
    pyrfu.mms.feeps_omni, pyrfu.mms.feeps_spin_avg

    """

    # Get all telescopes
    dataset_feeps = get_feeps_alleyes(tar_var, tint, mms_id, verbose,
                                      data_path)

    # Remove bad eyes and bad energy channels (lowest)
    dataset_feeps_washed = feeps_remove_bad_data(dataset_feeps)

    # Separate last channel
    dataset_feeps_clean, feeps_500kev = feeps_split_integral_ch(
        dataset_feeps_washed)

    # Remove sunlight contamination
    dataset_feeps_clean_sun_removed = feeps_remove_sun(dataset_feeps_clean)

    # Compute omni.directional spectrum
    spec_feeps_omni = feeps_omni(dataset_feeps_clean_sun_removed)

    if spin_avg:
        spec_feeps_omni = feeps_spin_avg(spec_feeps_omni,
                                         dataset_feeps.spinsectnum)

    return spec_feeps_omni
