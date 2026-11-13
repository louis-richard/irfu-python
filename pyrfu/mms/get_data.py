#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import os
import posixpath
import time
from typing import Mapping, Optional, Tuple, Union

# 3rd party imports
import requests
from botocore.exceptions import (
    ClientError,
    IncompleteReadError,
    ReadTimeoutError,
    ResponseStreamingError,
)
from requests.sessions import Session
from xarray import DataArray, Dataset

# Local imports
from pyrfu.mms import _cache
from pyrfu.mms.db_init import _read_config
from pyrfu.mms.get_dist import get_dist
from pyrfu.mms.get_ts import get_ts
from pyrfu.mms.list_files import list_files
from pyrfu.mms.list_files_aws import _s3_resource, list_files_aws
from pyrfu.mms.list_files_sdc import SDC_TIMEOUT, _login_lasp, list_files_sdc
from pyrfu.mms.tokenize import tokenize
from pyrfu.pyrf.dist_append import dist_append
from pyrfu.pyrf.time_clip import _time_interval
from pyrfu.pyrf.ts_append import ts_append

logger = logging.getLogger(__name__)

# Attempts to read a file from S3: the reads of the body of the response (a
# connection reset or timed out while the file is received) are not retried by
# botocore, which retries the requests only
_AWS_READ_ATTEMPTS = 3
_AWS_READ_ERRORS = (IncompleteReadError, ReadTimeoutError, ResponseStreamingError)


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


def _listing_key(source: str, tint: list, mms_id: str, var: Mapping) -> dict:
    r"""Key of the listing of the files of `var` in `tint` in the cache: the
    source and its settings (bucket, SDC rights and user), the spacecraft, the
    instrument, rate, level and data type, and the time interval."""
    config = _read_config()
    where = config["aws"] if source == "aws" else config["sdc"]
    product = {k: var.get(k) or "" for k in ["inst", "tmmode", "lev", "dtype"]}

    return {
        "source": source,
        "where": where,
        "mms_id": mms_id,
        **product,
        "tint": [str(t) for t in tint],
    }


def _remote_files(source: str, tint: list, mms_id: str, var: Mapping) -> list:
    r"""Files of `var` in `tint` on the SDC or AWS, as items of a listing of the
    cache (see :class:`pyrfu.mms._cache.RemoteFile`)."""
    if source == "aws":
        return [
            {
                "source": "aws",
                "path": _cache.mms_path(
                    mms_id, var, posixpath.basename(file["full_name"])
                ),
                "size": _cache.file_size(file["file_size"]),
                "bucket": file["s3_obj"].bucket_name,
                "key": file["s3_obj"].key,
            }
            for file in list_files_aws(tint, mms_id, var)
        ]

    return [
        {
            "source": "sdc",
            "path": _cache.mms_path(mms_id, var, file["file_name"]),
            "size": _cache.file_size(file.get("size")),
            "url": file["url"],
        }
        for file in list_files_sdc(tint, mms_id, var)
    ]


def _resolve_source(source: Optional[str] = "default") -> str:
    r"""Resource to fetch the data from: `source` ("local", "sdc" or "aws", in any
    case), or the default resource of the MMS configuration
    (`pyrfu.mms.MMS_CFG_PATH`) if it is empty or "default"."""
    if not source or source.lower() == "default":
        return _read_config()["default"]

    if source.lower() in ["local", "sdc", "aws"]:
        return source.lower()

    raise ValueError("Invalid source. Must be one of 'default', 'local', 'sdc', 'aws'")


def _list_files_sources(
    source: str,
    tint: list[str],
    mms_id: str,
    var: Mapping[str, Union[str, int]],
    data_path: Optional[str] = "",
):
    # Time interval in the forms of time_clip (only a list of str was
    # listed), as datetime64; the cache key of the listing is then the same
    # for all the forms
    tint = _time_interval(tint)

    if source == "local":
        file_names = list_files(tint, mms_id, var, data_path)
        sdc_session, headers = None, {}
    elif source in ["sdc", "aws"] and _cache.enabled():
        # Files of the cache listing (the SDC login only if a file is read)
        items = _cache.listing(
            _listing_key(source, tint, mms_id, var),
            lambda: _remote_files(source, tint, mms_id, var),
        )
        file_names = [_cache.RemoteFile(**item) for item in items]
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
    if isinstance(file_name, _cache.RemoteFile):
        return _cache.read(
            file_name, lambda: _download(file_name, sdc_session, headers)
        )

    if source == "local":
        file_path = os.path.normpath(file_name)
        with open(file_path, "rb") as file:
            file_content = file.read()
    elif source == "sdc":
        try:
            response = sdc_session.get(file_name, timeout=SDC_TIMEOUT, headers=headers)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            file_content = response.content
        except requests.RequestException:
            logger.error("Error retrieving file from %s", file_name)
            raise
    elif source == "aws":
        file_content = _read_s3_object(file_name)
    else:
        raise NotImplementedError(f"Resource {source} is not yet implemented!!")

    return file_content


def _get_s3_body(s3_object, label: str) -> bytes:
    r"""Content of an S3 object (S3 errors logged and raised)."""
    try:
        return s3_object.get()["Body"].read()
    except ClientError as err:
        error = err.response.get("Error", {})
        response_meta = err.response.get("ResponseMetadata", {})
        logger.error(
            "Error retrieving %s from S3: %s (%s, request ID %s, HTTP code %s)",
            label,
            error.get("Message"),
            error.get("Code"),
            response_meta.get("RequestId"),
            response_meta.get("HTTPStatusCode"),
        )
        raise


def _read_s3_object(s3_object) -> bytes:
    r"""Content of an S3 object, the request made again (after 1 s, then 2 s)
    if the connection is reset or times out while the file is received."""
    label = getattr(s3_object, "key", s3_object)

    for attempt in range(1, _AWS_READ_ATTEMPTS):
        try:
            return _get_s3_body(s3_object, label)
        except _AWS_READ_ERRORS as err:
            logger.warning(
                "Reading %s from S3 failed (%s), trying again (%d/%d)",
                label,
                err,
                attempt + 1,
                _AWS_READ_ATTEMPTS,
            )
            time.sleep(attempt)

    try:
        return _get_s3_body(s3_object, label)
    except _AWS_READ_ERRORS as err:
        logger.error("Error reading %s from S3: %s", label, err)
        raise


def _file_label(file_name) -> str:
    r"""Name of a file of the listings, for the messages: the base name of a
    local path, of the key of an S3 object (the messages printed the object),
    of a file of the cache, or the file of an SDC download URL."""
    if isinstance(file_name, _cache.RemoteFile):
        return str(file_name)

    name = str(getattr(file_name, "key", file_name))

    if "?file=" in name:
        return name.rsplit("?file=", 1)[1]

    return posixpath.basename(name.replace(os.sep, "/"))


def _download(remote, sdc_session=None, headers=None) -> bytes:
    r"""Content of a file of the cache listing, from AWS or the SDC (logged in
    if no session is given)."""
    if remote.source == "aws":
        return _get_file_content_sources(
            "aws", _s3_resource().Object(remote.bucket, remote.key)
        )

    if sdc_session is None:
        sdc_session, headers, _ = _login_lasp()

    return _get_file_content_sources("sdc", remote.url, sdc_session, headers)


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

    ASPOC: ionc_aspoc_srvy_l2

    DFG: b_bcs_dfg_srvy_l2pre, b_dmpa_dfg_srvy_l2pre, b_gse_dfg_srvy_l2pre,
    b_gsm_dfg_srvy_l2pre

    DSP: bpsd_dsp_fast_l2, bpsd_dsp_slow_l2, bpsdscm1_dsp_fast_l2, bpsdscm1_dsp_slow_l2,
    bpsdscm2_dsp_fast_l2, bpsdscm2_dsp_slow_l2, bpsdscm3_dsp_fast_l2,
    bpsdscm3_dsp_slow_l2, epsd_dsp_fast_l2, epsd_dsp_slow_l2, epsdx_dsp_fast_l2,
    epsdx_dsp_slow_l2, epsdy_dsp_fast_l2, epsdy_dsp_slow_l2, epsdz_dsp_fast_l2,
    epsdz_dsp_slow_l2

    EDI: e_dsl_edi_srvy_l2, e_gse_edi_srvy_l2, e_gsm_edi_srvy_l2,
    vdrift_dsl_edi_srvy_l2, vdrift_gse_edi_srvy_l2, vdrift_gsm_edi_srvy_l2

    EDP: bitmaske_edp_brst_l2, bitmaske_edp_fast_l2, bitmaske_edp_slow_l2,
    bitmaskv_edp_brst_l2, bitmaskv_edp_fast_l2, bitmaskv_edp_slow_l2,
    e2d_dsl_edp_brst_l2pre, e2d_dsl_edp_brst_ql, e2d_dsl_edp_fast_l2pre,
    e2d_dsl_edp_fast_ql, e_dsl_edp_brst_l2, e_dsl_edp_brst_l2pre, e_dsl_edp_brst_ql,
    e_dsl_edp_fast_l2, e_dsl_edp_fast_l2pre, e_dsl_edp_fast_ql, e_dsl_edp_slow_l2,
    e_dsl_edp_slow_l2pre, e_gse_edp_brst_l2, e_gse_edp_fast_l2, e_gse_edp_slow_l2,
    e_par_edp_brst_l2, e_par_edp_brst_l2pre, e_par_edp_fast_l2, e_par_edp_fast_l2pre,
    e_par_edp_slow_l2, e_par_edp_slow_l2pre, e_ssc_edp_brst_l2a, e_ssc_edp_fast_l2a,
    e_ssc_edp_slow_l2a, erre_edp_fast_l2, erre_edp_slow_l2, hmfe_dsl_edp_brst_l2,
    hmfe_par_edp_brst_l2, phase_edp_fast_l2a, phase_edp_slow_l2a, qualitye_edp_brst_l2,
    qualitye_edp_fast_l2, qualitye_edp_slow_l2, qualityv_edp_brst_l2,
    qualityv_edp_fast_l2, qualityv_edp_slow_l2, sdev12_edp_fast_l2a,
    sdev12_edp_slow_l2a, sdev34_edp_fast_l2a, sdev34_edp_slow_l2a, v6_edp_brst_l2,
    v6_edp_fast_l2, v6_edp_slow_l2, v_edp_brst_l2, v_edp_fast_l2, v_edp_fast_sitl,
    v_edp_slow_l2, v_edp_slow_sitl, vpsp_edp_brst_l2, vpsp_edp_fast_l2, vpsp_edp_slow_l2

    FGM: b_bcs_fgm_brst_l2, b_bcs_fgm_srvy_l2, b_dmpa_fgm_brst_l2, b_dmpa_fgm_srvy_l2,
    b_gse_fgm_brst_l2, b_gse_fgm_srvy_l2, b_gsm_fgm_brst_l2, b_gsm_fgm_srvy_l2,
    flag_fgm_brst_l2, flag_fgm_srvy_l2, r_gse_fgm_brst_l2, r_gse_fgm_srvy_l2,
    r_gsm_fgm_brst_l2, r_gsm_fgm_srvy_l2

    FPI: defbgi_fpi_brst_l2, defbgi_fpi_fast_l2, defe_fpi_brst_l2, defe_fpi_fast_l2,
    defe_fpi_fast_ql, defi_fpi_brst_l2, defi_fpi_fast_l2, errne_fpi_brst_l2,
    errne_fpi_fast_l2, errni_fpi_brst_l2, errni_fpi_fast_l2, errpe_fpi_brst_l2,
    errpe_fpi_fast_l2, errpi_fpi_brst_l2, errpi_fpi_fast_l2, errqe_fpi_brst_l2,
    errqe_fpi_fast_l2, errqi_fpi_brst_l2, errqi_fpi_fast_l2, errte_fpi_brst_l2,
    errte_fpi_fast_l2, errti_fpi_brst_l2, errti_fpi_fast_l2, errve_fpi_brst_l2,
    errve_fpi_fast_l2, errvi_fpi_brst_l2, errvi_fpi_fast_l2, flage_fpi_brst_l2,
    flage_fpi_fast_l2, flagi_fpi_brst_l2, flagi_fpi_fast_l2, nbgi_fpi_brst_l2,
    nbgi_fpi_fast_l2, ne_fpi_brst_l2, ne_fpi_fast_l2, ne_fpi_fast_ql,
    nexthighe_fpi_brst_l2, nexthighe_fpi_fast_l2, nexthighi_fpi_brst_l2,
    nexthighi_fpi_fast_l2, nextlowe_fpi_brst_l2, nextlowe_fpi_fast_l2,
    nextlowi_fpi_brst_l2, nextlowi_fpi_fast_l2, ni_fpi_brst_l2, ni_fpi_fast_l2,
    ni_fpi_fast_ql, padhighene_fpi_brst_l2, padhighene_fpi_fast_l2,
    padlowene_fpi_brst_l2, padlowene_fpi_fast_l2, padmidene_fpi_brst_l2,
    padmidene_fpi_fast_l2, partidxe_fpi_brst_l2, partidxe_fpi_fast_l2,
    partidxi_fpi_brst_l2, partidxi_fpi_fast_l2, partne_fpi_brst_l2, partne_fpi_fast_l2,
    partni_fpi_brst_l2, partni_fpi_fast_l2, partpe_dbcs_fpi_brst_l2,
    partpe_dbcs_fpi_fast_l2, partpe_gse_fpi_brst_l2, partpe_gse_fpi_fast_l2,
    partpi_dbcs_fpi_brst_l2, partpi_dbcs_fpi_fast_l2, partpi_gse_fpi_brst_l2,
    partpi_gse_fpi_fast_l2, partte_dbcs_fpi_brst_l2, partte_dbcs_fpi_fast_l2,
    partte_gse_fpi_brst_l2, partte_gse_fpi_fast_l2, partti_dbcs_fpi_brst_l2,
    partti_dbcs_fpi_fast_l2, partti_gse_fpi_brst_l2, partti_gse_fpi_fast_l2,
    parttparae_fpi_brst_l2, parttparae_fpi_fast_l2, parttparai_fpi_brst_l2,
    parttparai_fpi_fast_l2, parttperpe_fpi_brst_l2, parttperpe_fpi_fast_l2,
    parttperpi_fpi_brst_l2, parttperpi_fpi_fast_l2, partve_dbcs_fpi_brst_l2,
    partve_dbcs_fpi_fast_l2, partve_gse_fpi_brst_l2, partve_gse_fpi_fast_l2,
    partvi_dbcs_fpi_brst_l2, partvi_dbcs_fpi_fast_l2, partvi_gse_fpi_brst_l2,
    partvi_gse_fpi_fast_l2, pbgi_fpi_brst_l2, pbgi_fpi_fast_l2, pde_fpi_brst_l2,
    pde_fpi_fast_l2, pderre_fpi_brst_l2, pderre_fpi_fast_l2, pderri_fpi_brst_l2,
    pderri_fpi_fast_l2, pdi_fpi_brst_l2, pdi_fpi_fast_l2, pe_dbcs_fpi_brst_l2,
    pe_dbcs_fpi_fast_l2, pe_dbcs_fpi_fast_ql, pe_gse_fpi_brst_l2, pe_gse_fpi_fast_l2,
    pe_gse_fpi_fast_ql, pi_dbcs_fpi_brst_l2, pi_dbcs_fpi_fast_l2, pi_dbcs_fpi_fast_ql,
    pi_gse_fpi_brst_l2, pi_gse_fpi_fast_l2, pi_gse_fpi_fast_ql, qe_dbcs_fpi_brst_l2,
    qe_dbcs_fpi_fast_l2, qe_gse_fpi_brst_l2, qe_gse_fpi_fast_l2, qi_dbcs_fpi_brst_l2,
    qi_dbcs_fpi_fast_l2, qi_gse_fpi_brst_l2, qi_gse_fpi_fast_l2, ste_dbcs_fpi_brst_l2,
    ste_dbcs_fpi_fast_l2, ste_gse_fpi_brst_l2, ste_gse_fpi_fast_l2,
    sti_dbcs_fpi_brst_l2, sti_dbcs_fpi_fast_l2, sti_gse_fpi_brst_l2,
    sti_gse_fpi_fast_l2, te_dbcs_fpi_brst_l2, te_dbcs_fpi_fast_l2, te_dbcs_fpi_fast_ql,
    te_gse_fpi_brst_l2, te_gse_fpi_fast_l2, te_gse_fpi_fast_ql, ti_dbcs_fpi_brst_l2,
    ti_dbcs_fpi_fast_l2, ti_gse_fpi_brst_l2, ti_gse_fpi_fast_l2, tparae_fpi_brst_l2,
    tparae_fpi_fast_l2, tparai_fpi_brst_l2, tparai_fpi_fast_l2, tperpe_fpi_brst_l2,
    tperpe_fpi_fast_l2, tperpi_fpi_brst_l2, tperpi_fpi_fast_l2, ve_dbcs_fpi_brst_l2,
    ve_dbcs_fpi_fast_l2, ve_dbcs_fpi_fast_ql, ve_gse_fpi_brst_l2, ve_gse_fpi_fast_l2,
    ve_gse_fpi_fast_ql, vi_dbcs_fpi_brst_l2, vi_dbcs_fpi_fast_l2, vi_dbcs_fpi_fast_ql,
    vi_gse_fpi_brst_l2, vi_gse_fpi_fast_l2, vi_gse_fpi_fast_ql

    FSM: b_gse_fsm_brst_l3

    HPCA: azimuth_hpca_brst_l2, azimuth_hpca_srvy_l2, b_gse_hpca_brst_l2,
    b_gse_hpca_srvy_l2, b_gsm_hpca_brst_l2, b_gsm_hpca_srvy_l2, dpfheplus_hpca_brst_l2,
    dpfheplus_hpca_srvy_l2, dpfheplusplus_hpca_brst_l2, dpfheplusplus_hpca_srvy_l2,
    dpfhplus_hpca_brst_l2, dpfhplus_hpca_srvy_l2, dpfoplus_hpca_brst_l2,
    dpfoplus_hpca_srvy_l2, nheplus_hpca_brst_l2, nheplus_hpca_srvy_l2,
    nheplusplus_hpca_brst_l2, nheplusplus_hpca_srvy_l2, nhplus_hpca_brst_l2,
    nhplus_hpca_srvy_l2, noplus_hpca_brst_l2, noplus_hpca_srvy_l2,
    pheplus_dbcs_hpca_brst_l2, pheplus_dbcs_hpca_srvy_l2, pheplusplus_dbcs_hpca_brst_l2,
    pheplusplus_dbcs_hpca_srvy_l2, phplus_dbcs_hpca_brst_l2, phplus_dbcs_hpca_srvy_l2,
    poplus_dbcs_hpca_brst_l2, poplus_dbcs_hpca_srvy_l2, saz_hpca_brst_l2,
    saz_hpca_srvy_l2, theplus_dbcs_hpca_brst_l2, theplus_dbcs_hpca_srvy_l2,
    theplusplus_dbcs_hpca_brst_l2, theplusplus_dbcs_hpca_srvy_l2,
    thplus_dbcs_hpca_brst_l2, thplus_dbcs_hpca_srvy_l2, toplus_dbcs_hpca_brst_l2,
    toplus_dbcs_hpca_srvy_l2, tparaheplus_hpca_brst_l2, tparaheplus_hpca_srvy_l2,
    tparaheplusplus_hpca_brst_l2, tparaheplusplus_hpca_srvy_l2, tparahplus_hpca_brst_l2,
    tparahplus_hpca_srvy_l2, tparaoplus_hpca_brst_l2, tparaoplus_hpca_srvy_l2,
    tperpheplus_hpca_brst_l2, tperpheplus_hpca_srvy_l2, tperpheplusplus_hpca_brst_l2,
    tperpheplusplus_hpca_srvy_l2, tperphplus_hpca_brst_l2, tperphplus_hpca_srvy_l2,
    tperpoplus_hpca_brst_l2, tperpoplus_hpca_srvy_l2, tsheplus_hpca_brst_l2,
    tsheplus_hpca_srvy_l2, tsheplusplus_hpca_brst_l2, tsheplusplus_hpca_srvy_l2,
    tshplus_hpca_brst_l2, tshplus_hpca_srvy_l2, tsoplus_hpca_brst_l2,
    tsoplus_hpca_srvy_l2, vheplus_dbcs_hpca_brst_l2, vheplus_dbcs_hpca_srvy_l2,
    vheplus_gsm_hpca_brst_l2, vheplus_gsm_hpca_srvy_l2, vheplusplus_dbcs_hpca_brst_l2,
    vheplusplus_dbcs_hpca_srvy_l2, vheplusplus_gsm_hpca_brst_l2,
    vheplusplus_gsm_hpca_srvy_l2, vhplus_dbcs_hpca_brst_l2, vhplus_dbcs_hpca_srvy_l2,
    vhplus_gsm_hpca_brst_l2, vhplus_gsm_hpca_srvy_l2, voplus_dbcs_hpca_brst_l2,
    voplus_dbcs_hpca_srvy_l2, voplus_gsm_hpca_brst_l2, voplus_gsm_hpca_srvy_l2

    MEC: dst_mec_brst_l2, dst_mec_srvy_l2, eclipseearth_mec_brst_l2,
    eclipseearth_mec_srvy_l2, eclipsemoon_mec_brst_l2, eclipsemoon_mec_srvy_l2,
    kp_mec_brst_l2, kp_mec_srvy_l2, ldip_mec_brst_l2, ldip_mec_srvy_l2,
    mlat_mec_brst_l2, mlat_mec_srvy_l2, mlt_mec_brst_l2, mlt_mec_srvy_l2,
    quat_bcs_mec_brst_l2, quat_bcs_mec_srvy_l2, quat_dbcs_mec_brst_l2,
    quat_dbcs_mec_srvy_l2, quat_dmpa_mec_brst_l2, quat_dmpa_mec_srvy_l2,
    quat_dsl_mec_brst_l2, quat_dsl_mec_srvy_l2, quat_geo_mec_brst_l2,
    quat_geo_mec_srvy_l2, quat_gse2000_mec_brst_l2, quat_gse2000_mec_srvy_l2,
    quat_gse_mec_brst_l2, quat_gse_mec_srvy_l2, quat_gsm_mec_brst_l2,
    quat_gsm_mec_srvy_l2, quat_sm_mec_brst_l2, quat_sm_mec_srvy_l2,
    quat_smpa_mec_brst_l2, quat_smpa_mec_srvy_l2, quat_ssl_mec_brst_l2,
    quat_ssl_mec_srvy_l2, r_gse_mec_brst_l2, r_gse_mec_srvy_l2, r_gsm_mec_brst_l2,
    r_gsm_mec_srvy_l2, r_sm_mec_brst_l2, r_sm_mec_srvy_l2, vel_gse_mec_brst_l2,
    vel_gse_mec_srvy_l2, vel_gsm_mec_brst_l2, vel_gsm_mec_srvy_l2

    SCM: b_gse_scm_brst_l2, b_gse_scm_srvy_l2, bschb_gse_scm_brst_l2

    Parameters
    ----------
    var_str : str
        Key of the target variable (use mms.get_data() to see keys.).
    tint : array_like or xarray.DataArray
        Time interval: the start and stop times as ISO 8601 str, datetime64
        or datetime.datetime, or a time series (its first and last times), as
        `pyrfu.pyrf.time_clip`.
    mms_id : str or int
        Index of the target spacecraft.
    verbose : bool, Optional
        Set to True to follow the loading. Default is True.
    data_path : str, Optional
        Local path of MMS data. Default uses the local path of the MMS
        configuration (`pyrfu.mms.MMS_CFG_PATH`, set with `mms.db_init`).
    source : {"local", "sdc", "aws", "default"}, Optional
        Resource to fetch data from, in any case. Default (or "default") uses
        the default resource of the MMS configuration (`pyrfu.mms.MMS_CFG_PATH`,
        set with `mms.db_init`).

    Returns
    -------
    out : DataArray or Dataset
        Time series of the target variable of measured by the target
        spacecraft over the selected time interval.

    Raises
    ------
    FileNotFoundError
        If no file of the variable covers the time interval.
    ValueError
        If `source` is not "local", "sdc", "aws" or "default" (it was
        NotImplementedError, unlike the other readers).

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

    # The source in any case, or the default of the MMS configuration if empty
    # or "default"; ValueError for an unknown one, as db_get_ts,
    # db_get_variable, get_eis_allt and load_ancillary (it was
    # NotImplementedError)
    source = _resolve_source(source)

    file_names, sdc_session, headers = _list_files_sources(
        source, tint, mms_id, var, data_path
    )

    out = None

    if not file_names:
        raise FileNotFoundError(f"No files found for {var_str} in {source}")

    if verbose:
        logger.info("Loading %s...", cdf_name)

    for file_name in file_names:
        file_content = _get_file_content_sources(
            source, file_name, sdc_session, headers
        )

        if "-dist" in var["dtype"]:
            out = dist_append(out, get_dist(file_content, cdf_name, tint))

        else:
            out = ts_append(out, get_ts(file_content, cdf_name, tint))

    return out
