#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import logging
import os

# 3rd party imports
import numpy as np
import pycdfpp

from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.time_clip import time_clip
from ..pyrf.ts_skymap import ts_skymap
from .db_get_ts import db_get_ts

# Local imports
from .db_init import MMS_CFG_PATH
from .get_data import get_data

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


def remove_edist_background(vdf, n_sec: float = 0.0, n_art: float = -1.0):
    r"""Remove secondary photoelectrons from electron distribution function
    according to [1]_.

    Parameters
    ----------
    vdf : xarray.Dataset
        Measured electron velocity distribution function.
    n_sec : float, Optional
        Artificial secondary electron density (isotropic). Default is 0.
    n_art : float, Optional
        Artificial photoelectron density (sun-angle dependant),
        Default is ephoto_scale from des-emoms GlobalAttributes.

    Returns
    -------
    vdf_new : xarray.Dataset
        Electron VDF with photoelectrons removed.
    vdf_bkg : xarray.Dataset
        Photoelectron VDF.
    photoe_scle : float
        Artificial photoelectron and secondary electron density

    References
    ----------
    .. [1]  Gershman, D. J., Avanov, L. A., Boardsen,S. A., Dorelli, J. C.,
            Gliese, U., Barrie, A. C.,... Pollock, C. J. (2017). Spacecraft
            and instrument photoelectrons measured by the dual electron
            spectrometers on MMS. Journal of Geophysical Research:Space
            Physics,122, 11,548–11,558. https://doi.org/10.1002/2017JA024518

    """

    # Time interval of VDF
    tint = list(datetime642iso8601(vdf.time.data[[0, -1]]))

    # Get spacecraft index from VDF metadata
    mms_id = vdf.data.attrs["CATDESC"].split(" ")[0].lower()
    mms_id = int(mms_id[-1])

    # Get data sample rate from VDF metadata
    if "brst" in vdf.data.attrs["FIELDNAM"].lower():
        data_rate = "brst"
        logging.info("Burst resolution data is used")
    elif "fast" in vdf.data.attrs["FIELDNAM"].lower():
        data_rate = "fast"
        logging.info("Fast resolution data is used")
    else:
        raise TypeError("Could not identify if data is fast or burst.")

    vdf_tmp = time_clip(vdf, tint)
    vdf_new = np.zeros_like(vdf_tmp.data.data)
    vdf_bkg = np.zeros_like(vdf_tmp.data.data)

    dataset_name = f"mms{mms_id}_fpi_{data_rate}_l2_des-dist"
    startdelphi_count = db_get_ts(
        dataset_name,
        f"mms{mms_id}_des_startdelphi_count_{data_rate}",
        tint,
        verbose=False,
    )

    # Load the elctron number density to get the name of the photoelectron
    # model file, and the photoelectron scaling factor
    n_e = get_data(f"ne_fpi_{data_rate}_l2", tint, mms_id, verbose=False)

    photoe_scle = n_e.attrs["GLOBAL"]["Photoelectron_model_scaling_factor"]
    photoe_scle = float(photoe_scle)

    # Load the model internal photoelectrons
    bkg_fname = n_e.attrs["GLOBAL"]["Photoelectron_model_filenames"]

    # Read the current version of the MMS configuration file
    with open(MMS_CFG_PATH, "r", encoding="utf-8") as fs:
        config = json.load(fs)

    data_path = os.path.normpath(config["local"])

    bkg_fname = os.path.join(data_path, "models", "fpi", bkg_fname)

    f = pycdfpp.load(bkg_fname)

    if data_rate.lower() == "brst":
        prefs = ["mms_des_bgdist_p0", "mms_des_bgdist_p0"]
    else:
        prefs = ["mms_des_bgdist", "mms_des_bgdist"]

    vdf_bkg01 = [
        np.transpose(f[f"{prefs[0]}_{data_rate}"].values, [0, 3, 1, 2]),
        np.transpose(f[f"{prefs[1]}_{data_rate}"].values, [0, 3, 1, 2]),
    ]

    # Overwrite fraction of photoelectron if provided by user.
    if n_art > 0:
        n_photo = n_art
    else:
        n_photo = photoe_scle

    for i, _ in enumerate(vdf.time.data):
        istartdelphi_count = startdelphi_count.data[i]
        iebgdist = int(np.fix(istartdelphi_count / 16))

        esteptable_idx = int(vdf.attrs["esteptable"][i])
        vdf_bkg_tmp_data = vdf_bkg01[esteptable_idx][iebgdist, ...]
        vdf_bkg_tmp = n_photo * vdf_bkg_tmp_data

        if n_sec > 0:
            vdf_bkg_av = np.nanmean(vdf_bkg_tmp_data, axis=2)
            vdf_bkg_av = np.tile(vdf_bkg_av, (vdf_bkg_tmp_data.shape[2], 1, 1))
            vdf_bkg_av = np.transpose(vdf_bkg_av, [1, 2, 0])

            vdf_bkg_tmp += n_sec * vdf_bkg_av

        vdf_new_tmp = vdf.data.data[i, ...] - vdf_bkg_tmp
        vdf_new_tmp[vdf_new_tmp < 0] = 0.0
        vdf_bkg_tmp[vdf_bkg_tmp < 0] = 0.0
        vdf_new[i, ...] = vdf_new_tmp
        vdf_bkg[i, ...] = vdf_bkg_tmp

    # Construct the new VDFs
    glob_attrs = vdf.attrs
    vdf_attrs = vdf.data.attrs
    coords_attrs = {k: vdf[k].attrs for k in ["time", "energy", "phi", "theta"]}

    vdf_new = ts_skymap(
        vdf.time.data,
        vdf_new,
        vdf.energy.data,
        vdf.phi.data,
        vdf.theta.data,
        attrs=vdf_attrs,
        coords_attrs=coords_attrs,
        glob_attrs=glob_attrs,
    )
    vdf_new.attrs = vdf.attrs
    vdf_bkg = ts_skymap(
        vdf.time.data,
        vdf_bkg,
        vdf.energy.data,
        vdf.phi.data,
        vdf.theta.data,
        attrs=vdf_attrs,
        coords_attrs=coords_attrs,
        glob_attrs=glob_attrs,
    )
    vdf_bkg.attrs = vdf.attrs

    return vdf_new, vdf_bkg, photoe_scle
