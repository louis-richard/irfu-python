#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import os

# 3rd party imports
import numpy as np

from cdflib import cdfread

# Local imports
from .db_get_ts import db_get_ts
from .get_data import get_data
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.time_clip import time_clip
from ..pyrf.ts_skymap import ts_skymap

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.3.26"
__status__ = "Prototype"


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
    mms_id = vdf.attrs["CATDESC"].split(" ")[0].lower()
    mms_id = int(mms_id[-1])

    vdf_tmp = time_clip(vdf, tint)
    vdf_new = np.zeros_like(vdf_tmp.data.data)
    vdf_bkg = np.zeros_like(vdf_tmp.data.data)

    dataset_name = f"mms{mms_id}_fpi_brst_l2_des-dist"
    startdelphi_count = db_get_ts(
        dataset_name,
        f"mms{mms_id}_des_startdelphi_count_brst",
        tint,
        verbose=False,
    )

    # Load the elctron number density to get the name of the photoelectron
    # model file, and the photoelectron scaling factor
    n_e = get_data("ne_fpi_brst_l2", tint, mms_id, verbose=False)

    photoe_scle = n_e.attrs["Photoelectron_model_scaling_factor"]
    photoe_scle = float(photoe_scle)

    # Load the model internal photoelectrons
    bkg_fname = n_e.attrs["Photoelectron_model_filenames"]

    # Check path
    # Guess data path from CDF attributes
    data_path = str(vdf.attrs["CDF"]).split(f"mms{mms_id}/fpi", maxsplit=1)[0]

    # Check if path exists if not use the default
    if not os.path.exists(data_path):
        pkg_path = os.path.dirname(os.path.abspath(__file__))

        # Read the current version of the MMS configuration file
        with open(os.path.join(pkg_path, "config.json"), "r", encoding="utf-8") as fs:
            config = json.load(fs)

        data_path = os.path.normpath(config["local_data_dir"])
    else:
        data_path = os.path.normpath(data_path)

    bkg_fname = os.path.join(data_path, "models", "fpi", bkg_fname)

    vdf_bkg01 = [None, None]
    with cdfread.CDF(bkg_fname) as f:
        vdf_bkg01[0] = f.varget("mms_des_bgdist_p0_brst")
        vdf_bkg01[0] = np.transpose(vdf_bkg01[0], [0, 3, 1, 2])
        vdf_bkg01[1] = f.varget("mms_des_bgdist_p1_brst")
        vdf_bkg01[1] = np.transpose(vdf_bkg01[1], [0, 3, 1, 2])

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
