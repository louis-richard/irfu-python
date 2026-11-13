#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import os
import posixpath
from typing import Optional

# 3rd party imports
import numpy as np
import pycdfpp
from botocore.exceptions import ClientError

# Local imports
from ..pyrf.datetime642iso8601 import datetime642iso8601
from ..pyrf.ts_skymap import ts_skymap
from .db_get_ts import db_get_ts
from .db_init import _read_config
from .get_data import _resolve_source, get_data
from .list_files_aws import _bucket_and_prefix, _s3_resource
from .list_files_sdc import LASP_PUBL, SDC_TIMEOUT, _login_lasp

logger = logging.getLogger(__name__)


def _models_url(lasp_url: str) -> str:
    r"""URL of the FPI model files for the SDC access level of `lasp_url`."""
    return lasp_url.replace("files/api/v1/", "data/models/fpi/")


def _load_bgdist_model(file_name: str, source: str, data_path: str):
    r"""Load the FPI photoelectron model file `file_name`.

    Parameters
    ----------
    file_name : str
        Name of the model file (Photoelectron_model_filenames).
    source : {"local", "sdc", "aws"}
        Resource to read the model from. For "aws", the model is read from
        ``models/fpi/`` in the MMS bucket (HelioCloud by default), or from the
        SDC if it is not there.
    data_path : str
        Path of the local MMS data (model in ``<data_path>/models/fpi/``).

    Returns
    -------
    pycdfpp.CDF
        Content of the model file.

    Raises
    ------
    FileNotFoundError
        If the local model file doesn't exist.

    """
    if source == "local":
        file_path = os.path.join(data_path, "models", "fpi", file_name)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"FPI model file {file_path} not found. Download it from "
                f"{_models_url(LASP_PUBL)}{file_name} or use source='sdc'."
            )

        return pycdfpp.load(file_path)

    if source == "aws":
        bucket_name, prefix = _bucket_and_prefix()
        key = posixpath.join(prefix, "models", "fpi", file_name)

        try:
            response = _s3_resource().Object(bucket_name, key).get()
            return pycdfpp.load(response["Body"].read())
        except ClientError as err:
            logger.warning(
                "FPI model file s3://%s/%s not available (%s), reading it from the SDC",
                bucket_name,
                key,
                err.response.get("Error", {}).get("Code"),
            )

    # Read the file from the SDC into memory
    sdc_session, headers, lasp_url = _login_lasp()

    response = sdc_session.get(
        _models_url(lasp_url) + file_name, headers=headers, timeout=SDC_TIMEOUT
    )
    response.raise_for_status()

    return pycdfpp.load(response.content)


def remove_edist_background(
    vdf, n_sec: float = 0.0, n_art: Optional[float] = None, source: str = ""
):
    r"""Remove secondary photoelectrons from electron distribution function
    according to [1]_.

    Parameters
    ----------
    vdf : xarray.Dataset
        Measured electron velocity distribution function.
    n_sec : float, Optional
        Artificial secondary electron density (isotropic): `n_sec` times the
        photoelectron model averaged over azimuth (NaN bins left out) is also
        removed. Default is 0.
    n_art : float, Optional
        Artificial photoelectron density (sun-angle dependant). 0 removes no
        photoelectrons. Default is None, which uses the photoelectron scaling
        factor from the des-moms global attributes.
    source : {"default", "local", "sdc", "aws"}, Optional
        Resource to fetch the data (spin phase, moments) and the photoelectron
        model from, in any case. The model is read from ``<local>/models/fpi/``
        for "local", from the SDC for "sdc", and from ``models/fpi/`` in the
        MMS bucket (HelioCloud by default) for "aws". Default (or "default")
        uses the default resource of the MMS configuration
        (`pyrfu.mms.MMS_CFG_PATH`, set with `mms.db_init`).

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

    if mms_id not in [1, 2, 3, 4]:
        raise ValueError(
            f"Invalid MMS spacecraft number {mms_id}. Must be 1, 2, 3 or 4."
        )

    # Get data sample rate from VDF metadata
    if "brst" in vdf.data.attrs["FIELDNAM"].lower():
        data_rate = "brst"
        logger.info("Burst resolution data is used")
    elif "fast" in vdf.data.attrs["FIELDNAM"].lower():
        data_rate = "fast"
        logger.info("Fast resolution data is used")
    else:
        raise TypeError("Could not identify if data is fast or burst.")

    # Resource to read the data and the photoelectron model from, as the other
    # readers ("default" raised ValueError)
    source = _resolve_source(source)

    dataset_name = f"mms{mms_id}_fpi_{data_rate}_l2_des-dist"
    startdelphi_count = db_get_ts(
        dataset_name,
        f"mms{mms_id}_des_startdelphi_count_{data_rate}",
        tint,
        verbose=False,
        source=source,
    )

    # Load the elctron number density to get the name of the photoelectron
    # model file, and the photoelectron scaling factor
    n_e = get_data(f"ne_fpi_{data_rate}_l2", tint, mms_id, verbose=False, source=source)

    photoe_scle = n_e.attrs["GLOBAL"]["Photoelectron_model_scaling_factor"]
    photoe_scle = float(photoe_scle)

    # Load the model internal photoelectrons
    bkg_fname = n_e.attrs["GLOBAL"]["Photoelectron_model_filenames"]
    data_path = os.path.normpath(_read_config()["local"])
    f = _load_bgdist_model(bkg_fname, source, data_path)

    # Burst: model for each energy table (steptable parity 0 and 1)
    if data_rate.lower() == "brst":
        prefs = ["mms_des_bgdist_p0", "mms_des_bgdist_p1"]
    else:
        prefs = ["mms_des_bgdist", "mms_des_bgdist"]

    vdf_bkg01 = [
        np.transpose(f[f"{prefs[0]}_{data_rate}"].values, [0, 3, 1, 2]),
        np.transpose(f[f"{prefs[1]}_{data_rate}"].values, [0, 3, 1, 2]),
    ]

    # Spin phase of each distribution (startdelphi_count), taken at its time
    # rather than by position, as it is read separately
    startdelphi_count = startdelphi_count.sel(time=vdf.time.data, method="nearest")
    d_t = np.abs(startdelphi_count.time.data - vdf.time.data)

    if len(vdf.time) > 1 and np.max(d_t) > np.median(np.diff(vdf.time.data)) / 2:
        raise ValueError("startdelphi_count not available at the times of vdf")

    # Model record with the spin phase closest to that of each distribution
    # (MMS CMAD, FPI photoelectron model; about floor(startdelphi_count / 16))
    model_counts = f[f"mms_des_startdelphi_counts_{data_rate}"].values.astype(int)
    idx_model = np.argmin(
        np.abs(startdelphi_count.data.astype(int)[:, None] - model_counts[None, :]),
        axis=1,
    )

    # Overwrite fraction of photoelectron if provided by user (0 removes no
    # photoelectrons; None or negative uses the value from the moments file).
    if n_art is not None and n_art >= 0:
        n_photo = n_art
    else:
        n_photo = photoe_scle

    # The background of each (energy table, model record) once, given to the
    # distributions that use it (it was computed for each time): the model
    # scaled by n_photo, plus, for n_sec > 0, isotropic secondary electrons,
    # the model averaged over azimuth (axis 1 of energy, phi, theta), as
    # irfu-matlab (it was averaged over the polar angle)
    tables = np.asarray(vdf.attrs["esteptable"]).astype(int)
    pairs, which = np.unique(
        np.stack([tables, idx_model], axis=1), axis=0, return_inverse=True
    )

    backgrounds = []

    for table, record in pairs:
        model = vdf_bkg01[table][record, ...]
        background = n_photo * model

        if n_sec > 0:
            background = background + n_sec * np.nanmean(model, axis=1, keepdims=True)

        backgrounds.append(background)

    vdf_bkg = np.stack(backgrounds)[np.ravel(which)]

    # Background-subtracted distributions, negative values (and background)
    # set to 0
    vdf_new = vdf.data.data - vdf_bkg
    vdf_new[vdf_new < 0] = 0.0
    vdf_bkg[vdf_bkg < 0] = 0.0
    vdf_new = vdf_new.astype(vdf.data.data.dtype, copy=False)
    vdf_bkg = vdf_bkg.astype(vdf.data.data.dtype, copy=False)

    # Construct the new VDFs (copies, so the attributes of the input and of the
    # two outputs are independent)
    glob_attrs = dict(vdf.attrs)
    vdf_attrs = dict(vdf.data.attrs)
    coords_attrs = {k: dict(vdf[k].attrs) for k in ["time", "energy", "phi", "theta"]}

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
    vdf_new.attrs = dict(glob_attrs)
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
    vdf_bkg.attrs = dict(glob_attrs)

    return vdf_new, vdf_bkg, photoe_scle
