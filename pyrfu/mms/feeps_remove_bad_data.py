#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import os
import json
import datetime

# 3rd party imports
import numpy as np

# Local imports
from ..pyrf import iso86012datetime, datetime642iso8601

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def _bad_vars(bad_data):
    bad_vars_top = list(filter(lambda x: x not in [6, 7, 8], bad_data["top"]))
    bad_vars_bot = list(
        filter(lambda x: x not in [6, 7, 8], bad_data["bottom"]))

    bad_vars = [*[f"top-{x}" for x in bad_vars_top],
                *[f"bottom-{x}" for x in bad_vars_bot]]

    return bad_vars


def _bad_eyes(inp_dataset, bad_vars):
    inp_dataset_clean = inp_dataset.copy()

    for bad_var in bad_vars:
        if bad_var not in list(inp_dataset.keys()):
            continue

        inp_dataset_clean[bad_var].data[:] = np.nan

    return inp_dataset


def _bad_ch0(inp_dataset, bad_vars):
    inp_dataset_clean = inp_dataset.copy()

    for bad_var in bad_vars:
        if bad_var not in list(inp_dataset.keys()):
            continue
        # check if the energy table contains all nans
        energy = inp_dataset[inp_dataset[bad_var].dims[1]]
        if np.isnan(np.sum(energy.data)):
            continue

        inp_dataset_clean[bad_var].data[:, 0] = np.nan

    return inp_dataset_clean


def _bad_ch1(inp_dataset, bad_vars):
    inp_dataset_clean = inp_dataset.copy()

    for bad_var in bad_vars:
        if bad_var not in list(inp_dataset.keys()):
            continue
        # check if the energy table contains all nans
        energy = inp_dataset[inp_dataset[bad_var].dims[1]]
        if np.isnan(np.sum(energy.data)):
            continue

        inp_dataset_clean[bad_var].data[:, 0] = np.nan
        inp_dataset_clean[bad_var].data[:, 1] = np.nan

    return inp_dataset_clean


def _bad_ch2(inp_dataset, bad_vars):
    inp_dataset_clean = inp_dataset.copy()

    for bad_var in bad_vars:
        if bad_var not in list(inp_dataset.keys()):
            continue
        # check if the energy table contains all nans
        energy = inp_dataset[inp_dataset[bad_var].dims[1]]
        if np.isnan(np.sum(energy.data)):
            continue

        inp_dataset_clean[bad_var].data[:, 0] = np.nan
        inp_dataset_clean[bad_var].data[:, 1] = np.nan
        inp_dataset_clean[bad_var].data[:, 2] = np.nan

    return inp_dataset_clean


def feeps_remove_bad_data(inp_dataset):
    r"""This function removes bad eyes, bad lowest energy channels based on
    data from Drew Turner

    Parameters
    ----------
    inp_dataset : xarray.Dataset
        Dataset with all active telescopes data.

    Returns
    -------
    inp_dataaset_clean_all : xarray.Dataset
        Dataset with all active telescopes data where bad eyes and lab lowest
        energy channels are set to NaN.

    """

    mms_id = inp_dataset.attrs["mmsId"]

    root_path = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(root_path, "feeps_bad_data.json")) as file:
        feeps_bad_data = json.load(file)

    bad_data_table = feeps_bad_data["bad_data_table"]
    bad_ch0 = feeps_bad_data["bad_ch0"]
    bad_ch1 = feeps_bad_data["bad_ch1"]
    bad_ch2 = feeps_bad_data["bad_ch2"]

    # 1. BAD EYES
    # First, here is a list of the EYES that are bad, we need to make sure
    # these data are not usable (i.e., make all of the counts/rate/flux data
    # from these eyes NAN). These are for all modes, burst and survey:

    dates = [datetime.datetime.strptime(t_, "%Y-%m-%d").timestamp() for t_ in
             bad_data_table.keys()]

    t_data = iso86012datetime(datetime642iso8601(inp_dataset.time.data[0]))[0]
    closest_table_tm = np.argmin([t_ - t_data.timestamp() for t_ in dates])

    closest_table = list(bad_data_table.keys())[closest_table_tm]
    bad_data = bad_data_table[closest_table][f"mms{mms_id:d}"]

    bad_vars_eyes = _bad_vars(bad_data)

    inp_dataset_clean_eye = _bad_eyes(inp_dataset, bad_vars_eyes)

    # 2. BAD LOWEST E-CHANNELS
    # Next, these eyes have bad first channels (i.e., lowest energy channel,
    # E-channel 0 in IDL indexing). Again, these data (just the
    # counts/rate/flux from the lowest energy channel ONLY!!!) should be
    # hardwired to be NAN for all modes (burst and both types of survey).
    # The eyes not listed here or above are ok though... so once we do this,
    # we can actually start showing the data down to the lowest levels (~33
    # keV), meaning we'll have to adjust the hard-coded ylim settings in
    # SPEDAS and the SITL software:

    if t_data > iso86012datetime(["2019-05-01T00:00:00.000"])[0]:
        bad_data_ch0 = bad_ch0[">2019-05-01"][f"mms{mms_id}"]
        bad_data_ch1 = bad_ch1[">2019-05-01"][f"mms{mms_id}"]
        bad_data_ch2 = bad_ch2[">2019-05-01"][f"mms{mms_id}"]
    else:
        bad_data_ch0 = bad_ch0["<2019-05-01"][f"mms{mms_id}"]
        bad_data_ch1 = bad_ch1["<2019-05-01"][f"mms{mms_id}"]
        bad_data_ch2 = bad_ch2["<2019-05-01"][f"mms{mms_id}"]

    bad_vars_ch0 = _bad_vars(bad_data_ch0)
    inp_dataset_clean_ch0 = _bad_ch0(inp_dataset_clean_eye, bad_vars_ch0)

    bad_vars_ch1 = _bad_vars(bad_data_ch1)
    inp_dataset_clean_ch1 = _bad_ch1(inp_dataset_clean_ch0, bad_vars_ch1)

    bad_vars_ch2 = _bad_vars(bad_data_ch2)
    inp_dataset_clean_all = _bad_ch2(inp_dataset_clean_ch1, bad_vars_ch2)

    return inp_dataset_clean_all



