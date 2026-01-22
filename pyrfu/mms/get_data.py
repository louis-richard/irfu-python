#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import json
import logging
import os
from typing import Mapping, Optional, Tuple, Union

# 3rd party imports
import requests
from botocore.exceptions import ClientError
from requests.sessions import Session
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

# Local imports
from pyrfu.mms.db_init import MMS_CFG_PATH
from pyrfu.mms.get_dist import get_dist
from pyrfu.mms.get_ts import get_ts
from pyrfu.mms.list_files import list_files
from pyrfu.mms.list_files_aws import list_files_aws
from pyrfu.mms.list_files_sdc import _login_lasp, list_files_sdc
from pyrfu.mms.tokenize import tokenize
from pyrfu.pyrf.dist_append import dist_append
from pyrfu.pyrf.ts_append import ts_append
from pyrfu.pyrf.ttns2datetime64 import ttns2datetime64

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


def _var_and_cdf_name(
    var_str: str, mms_id: str
) -> Tuple[Mapping[str, Union[str, int]], str]:
    r"""Tokenize variable string `var_str` and returns the corresponding cdf name.

    Parameters
    ----------
    var_str : str
        Variable string.
    mms_id : str
        Spacecraft identifier.

    Returns
    -------
    var : dict
        Variable dictionary.
    cdf_name : str
        Corresponding cdf name.

    """
    var = tokenize(var_str)
    cdf_name = f"mms{mms_id}_{var['cdf_name']}"
    return var, cdf_name


def _check_times(inp: Union[DataArray, Dataset]) -> Union[DataArray, Dataset]:
    if inp.time.data.dtype == "int64":
        out = inp.assign_coords(time=ttns2datetime64(inp.time.data))
    else:
        out = inp
    return out


def _list_files_sources(
    source: str,
    tint: list[str],
    mms_id: str,
    var: Mapping[str, Union[str, int]],
    data_path: Optional[str] = "",
):
    if source == "local":
        file_names = list_files(tint, mms_id, var, data_path)
        sdc_session, headers = None, {}
    elif source == "sdc":
        file_names = [file.get("url") for file in list_files_sdc(tint, mms_id, var)]
        sdc_session, headers, _ = _login_lasp()
    elif source == "aws":
        file_names = [file.get("s3_obj") for file in list_files_aws(tint, mms_id, var)]
        sdc_session, headers = None, {}
    else:
        raise NotImplementedError(f"Resource {source} is not yet implemented!!")

    return file_names, sdc_session, headers


def _get_file_content_sources(
    source: str,
    file_name: str,
    sdc_session: Optional[Session] = None,
    headers: Optional[dict] = None,
) -> bytes:
    r"""Get file content from different sources.

    Parameters
    ----------
    source : str
        Source of the data.
    file_name : str
        File name.
    sdc_session : Session, Optional
        SDC session. Default is None.
    headers : dict, Optional
        Headers. Default is None.

    Returns
    -------
    file_content : bytes
        File content.

    """
    if source == "local":
        file_path = os.path.normpath(file_name)
        with open(file_path, "rb") as file:
            file_content = file.read()
    elif source == "sdc":
        try:
            response = sdc_session.get(file_name, timeout=None, headers=headers)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            file_content = response.content
        except requests.RequestException:
            logging.error("Error retrieving file from %s", file_name)
    elif source == "aws":
        try:
            response = file_name.get()
            file_content = response["Body"].read()
        except ClientError as err:
            if err.response["Error"]["Code"] == "InternalError":  # Generic error
                logging.error("Error Message: %s", err.response["Error"]["Message"])

                response_meta = err.response.get("ResponseMetadata")
                logging.error("Request ID: %s", response_meta.get("RequestId"))
                logging.error("Http code: %s", response_meta.get("HTTPStatusCode"))
            else:
                raise err
    else:
        raise NotImplementedError(f"Resource {source} is not yet implemented!!")

    return file_content


def get_data(
    var_str: str,
    tint: list,
    mms_id: Union[int, str],
    verbose: bool = True,
    data_path: str = "",
    source: str = "",
) -> Union[DataArray, Dataset]:
    r"""Load a variable `var_str`.

    AFG: b_bcs_afg_srvy_l2pre, b_dmpa_afg_srvy_l2pre, b_gse_afg_srvy_l2pre,
    b_gsm_afg_srvy_l2pre

    DFG: b_bcs_dfg_srvy_l2pre, b_dmpa_dfg_srvy_l2pre, b_gse_dfg_srvy_l2pre,
     b_gsm_dfg_srvy_l2pre

    EDP: e2d_dsl_edp_brst_l2pre, e2d_dsl_edp_brst_ql, e2d_dsl_edp_fast_l2pre,
    e2d_dsl_edp_fast_ql, e_dsl_edp_brst_l2, e_dsl_edp_brst_l2pre, e_dsl_edp_brst_ql,
    e_dsl_edp_fast_l2, e_dsl_edp_fast_l2pre, e_dsl_edp_fast_ql, e_dsl_edp_slow_l2,
    e_dsl_edp_slow_l2pre, e_gse_edp_brst_l2, e_gse_edp_fast_l2, e_gse_edp_slow_l2,
    e_ssc_edp_brst_l2a, e_ssc_edp_fast_l2a, e_ssc_edp_slow_l2a, hmfe_dsl_edp_brst_l2,
    phase_edp_fast_l2a, phase_edp_slow_l2a, sdev12_edp_fast_l2a, sdev12_edp_slow_l2a,
    sdev34_edp_fast_l2a, sdev34_edp_slow_l2a, v_edp_brst_l2, v_edp_fast_l2,
    v_edp_fast_sitl, v_edp_slow_l2, v_edp_slow_sitl

    FGM: b_bcs_fgm_brst_l2, b_bcs_fgm_srvy_l2, b_dmpa_fgm_brst_l2, b_dmpa_fgm_srvy_l2,
    b_gse_fgm_brst_l2, b_gse_fgm_srvy_l2, b_gsm_fgm_brst_l2, b_gsm_fgm_srvy_l2

    FPI: defbgi_fpi_brst_l2, defbgi_fpi_fast_l2, defe_fpi_brst_l2, defe_fpi_fast_l2,
    defe_fpi_fast_ql, defi_fpi_brst_l2, defi_fpi_fast_l2, nbgi_fpi_brst_l2,
    nbgi_fpi_fast_l2, ne_fpi_brst_l2, ne_fpi_fast_l2, ne_fpi_fast_ql, ni_fpi_brst_l2,
    ni_fpi_fast_l2, ni_fpi_fast_ql, partne_fpi_brst_l2, partne_fpi_fast_l2,
    partni_fpi_brst_l2, partni_fpi_fast_l2, partpe_gse_fpi_brst_l2,
    partpe_gse_fpi_fast_l2, partpi_gse_fpi_brst_l2, partpi_gse_fpi_fast_l2,
    partte_dbcs_fpi_brst_l2, partte_dbcs_fpi_fast_l2, partte_gse_fpi_brst_l2,
    partte_gse_fpi_fast_l2, partti_dbcs_fpi_brst_l2, partti_dbcs_fpi_fast_l2,
    partti_gse_fpi_brst_l2, partti_gse_fpi_fast_l2, parttparae_fpi_brst_l2,
    parttparai_fpi_brst_l2, parttparai_fpi_fast_l2, parttperpe_fpi_brst_l2,
    parttperpi_fpi_brst_l2, parttperpi_fpi_fast_l2, partve_dbcs_fpi_brst_l2,
    partve_dbcs_fpi_fast_l2, partve_gse_fpi_brst_l2, partve_gse_fpi_fast_l2,
    partvi_dbcs_fpi_brst_l2, partvi_dbcs_fpi_fast_l2, partvi_gse_fpi_brst_l2,
    partvi_gse_fpi_fast_l2, pbgi_fpi_brst_l2, pbgi_fpi_fast_l2, pde_fpi_brst_l2,
    pde_fpi_fast_l2, pderre_fpi_brst_l2, pderre_fpi_fast_l2, pderri_fpi_brst_l2,
    pderri_fpi_fast_l2, pdi_fpi_brst_l2, pdi_fpi_fast_l2, pe_dbcs_fpi_brst_l2,
    pe_dbcs_fpi_fast_l2, pe_dbcs_fpi_fast_ql, pe_gse_fpi_brst_l2, pe_gse_fpi_fast_l2,
    pe_gse_fpi_fast_ql, pi_dbcs_fpi_brst_l2, pi_dbcs_fpi_fast_l2, pi_dbcs_fpi_fast_ql,
    pi_gse_fpi_brst_l2, pi_gse_fpi_fast_l2, pi_gse_fpi_fast_ql, ste_dbcs_fpi_brst_l2,
    ste_dbcs_fpi_fast_l2, ste_gse_fpi_brst_l2, ste_gse_fpi_fast_l2,
    sti_dbcs_fpi_brst_l2, sti_dbcs_fpi_fast_l2, sti_gse_fpi_brst_l2,
    sti_gse_fpi_fast_l2, te_dbcs_fpi_brst_l2, te_dbcs_fpi_fast_l2,
    te_dbcs_fpi_fast_ql, te_gse_fpi_brst_l2, te_gse_fpi_fast_l2, te_gse_fpi_fast_ql,
    ti_dbcs_fpi_brst_l2, ti_dbcs_fpi_fast_l2, ti_gse_fpi_brst_l2, ti_gse_fpi_fast_l2,
    tparae_fpi_brst_l2, tparai_fpi_brst_l2, tparai_fpi_fast_l2, tperpe_fpi_brst_l2,
    tperpi_fpi_brst_l2, tperpi_fpi_fast_l2, ve_dbcs_fpi_brst_l2, ve_dbcs_fpi_fast_l2,
    ve_dbcs_fpi_fast_ql, ve_gse_fpi_brst_l2, ve_gse_fpi_fast_l2, ve_gse_fpi_fast_ql,
    vi_dbcs_fpi_brst_l2, vi_dbcs_fpi_fast_l2, vi_dbcs_fpi_fast_ql,
    vi_gse_fpi_brst_l2, vi_gse_fpi_fast_l2, vi_gse_fpi_fast_ql

    FSM: b_gse_fsm_brst_l3

    HPCA: azimuth_hpca_brst_l2, azimuth_hpca_srvy_l2, dpfheplus_hpca_brst_l2,
    dpfheplus_hpca_srvy_l2, dpfheplusplus_hpca_brst_l2, dpfheplusplus_hpca_srvy_l2,
    dpfhplus_hpca_brst_l2, dpfhplus_hpca_srvy_l2, dpfoplus_hpca_brst_l2,
    dpfoplus_hpca_srvy_l2, nheplus_hpca_brst_l2, nheplus_hpca_srvy_l2,
    nheplusplus_hpca_brst_l2, nheplusplus_hpca_srvy_l2, nhplus_hpca_brst_l2,
    nhplus_hpca_srvy_l2, noplus_hpca_brst_l2, noplus_hpca_srvy_l2, saz_hpca_brst_l2,
    saz_hpca_srvy_l2, theplus_dbcs_hpca_brst_l2, theplus_dbcs_hpca_srvy_l2,
    theplusplus_dbcs_hpca_brst_l2, theplusplus_dbcs_hpca_srvy_l2,
    thplus_dbcs_hpca_brst_l2, thplus_dbcs_hpca_srvy_l2, toplus_dbcs_hpca_brst_l2,
    toplus_dbcs_hpca_srvy_l2, tsheplus_hpca_brst_l2, tsheplus_hpca_srvy_l2,
    tsheplusplus_hpca_brst_l2, tsheplusplus_hpca_srvy_l2, tshplus_hpca_brst_l2,
    tshplus_hpca_srvy_l2, tsoplus_hpca_brst_l2, tsoplus_hpca_srvy_l2,
    vheplus_dbcs_hpca_brst_l2, vheplus_dbcs_hpca_srvy_l2, vheplus_gsm_hpca_brst_l2,
    vheplus_gsm_hpca_srvy_l2, vheplusplus_dbcs_hpca_brst_l2,
    vheplusplus_dbcs_hpca_srvy_l2, vheplusplus_gsm_hpca_brst_l2,
    vheplusplus_gsm_hpca_srvy_l2, vhplus_dbcs_hpca_brst_l2, vhplus_dbcs_hpca_srvy_l2,
    vhplus_gsm_hpca_brst_l2, vhplus_gsm_hpca_srvy_l2, voplus_dbcs_hpca_brst_l2,
    voplus_dbcs_hpca_srvy_l2, voplus_gsm_hpca_brst_l2, voplus_gsm_hpca_srvy_l2

    MEC: r_gse_mec_srvy_l2, r_gsm_mec_srvy_l2, r_gse_mec_brst_l2, r_gsm_mec_brst_l2,
    v_gse_mec_srvy_l2, v_gsm_mec_srvy_l2, v_gse_mec_brst_l2, v_gsm_mec_brst_l2

    SCM: b_gse_scm_brst_l2

    Parameters
    ----------
    var_str : str
        Key of the target variable (use mms.get_data() to see keys.).
    tint : list of str
        Time interval.
    mms_id : str or int
        Index of the target spacecraft.
    verbose : bool, Optional
        Set to True to follow the loading. Default is True.
    data_path : str, Optional
        Local path of MMS data. Default uses that provided in `pyrfu/mms/config.json`
    source: {"local", "sdc", "aws"}, Optional
        Ressource to fetch data from. Default uses default in `pyrfu/mms/config.json`

    Returns
    -------
    out : DataArray or Dataset
        Time series of the target variable of measured by the target
        spacecraft over the selected time interval.

    See also
    --------
    pyrfu.mms.get_ts : Read time series.
    pyrfu.mms.get_dist : Read velocity distribution function.

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint_brst = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Index of MMS spacecraft

    >>> ic = 1

    Load magnetic field from FGM

    >>> b_xyz = mms.get_data("b_gse_fgm_brst_l2", tint_brst, ic)

    """
    # Convert mms_id to string
    mms_id = str(mms_id)

    var, cdf_name = _var_and_cdf_name(var_str, mms_id)

    # Read the current version of the MMS configuration file
    with open(MMS_CFG_PATH, "r", encoding="utf-8") as fs:
        config = json.load(fs)

    source = source if source else config.get("default")

    file_names, sdc_session, headers = _list_files_sources(
        source, tint, mms_id, var, data_path
    )

    if not file_names:
        raise FileNotFoundError(f"No files found for {var_str} in {source}")

    if verbose:
        logging.info("Loading %s...", cdf_name)

    out = None

    for file_name in file_names:
        file_content = _get_file_content_sources(
            source, file_name, sdc_session, headers
        )

        if "-dist" in var["dtype"]:
            out = dist_append(out, get_dist(file_content, cdf_name, tint))

        else:
            out = ts_append(out, get_ts(file_content, cdf_name, tint))

    out = _check_times(out)

    if sdc_session:
        sdc_session.close()

    return out
