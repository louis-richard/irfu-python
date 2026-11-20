#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import fnmatch
import logging
import os
import posixpath
import time
from collections.abc import Callable, Mapping
from numbers import Integral
from typing import Any, cast, overload

# 3rd party imports
import requests
from botocore.exceptions import (
    ClientError,
    IncompleteReadError,
    ReadTimeoutError,
    ResponseStreamingError,
)
from pycdfpp import load
from requests.sessions import Session
from xarray import DataArray, Dataset

# Local imports
from pyrfu.mms._internal import cache as _cache
from pyrfu.mms.db_init import _read_config
from pyrfu.mms.get_dist import get_dist
from pyrfu.mms.get_ts import get_ts
from pyrfu.mms.list_files import list_files
from pyrfu.mms.list_files_aws import _s3_resource, list_files_aws
from pyrfu.mms.list_files_sdc import SDC_TIMEOUT, _login_lasp, list_files_sdc
from pyrfu.mms.tokenize import _tokenize, _warn_deprecated_key
from pyrfu.pyrf.dist_append import dist_append
from pyrfu.pyrf.time_clip import _time_interval
from pyrfu.pyrf.ts_append import ts_append

logger = logging.getLogger(__name__)

# Attempts to read a file from S3: the reads of the body of the response (a
# connection reset or timed out while the file is received) are not retried by
# botocore, which retries the requests only
_AWS_READ_ATTEMPTS = 3
_AWS_READ_ERRORS = (IncompleteReadError, ReadTimeoutError, ResponseStreamingError)


def _var_and_cdf_name(var_str: str, mms_id: str | int) -> tuple[Mapping[str, str], str]:
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

    var = _tokenize(var_str)
    cdf_name = f"mms{mms_id}_{var['cdf_name']}"
    return var, cdf_name


def _is_skymap(cdf_name: str) -> bool:
    r"""Whether the FPI variable `cdf_name` (with the spacecraft prefix) is a
    skymap distribution, read by `get_dist`: the distribution, its error, the
    modelled photoelectron distribution and the time offsets of the bins
    (dist, disterr, disterr_photo, steptimeoffsets).
    The other variables of the distribution files (e.g. avgf1counts) are time
    series."""

    return cdf_name.split("_")[2] in ("dist", "disterr", "steptimeoffsets")


def _read(reader, file_content, cdf_name, tint):
    r"""Time series or distribution `cdf_name` of a file, read with `reader`
    (`get_ts` or `get_dist`), or None if the file does not have the variable
    (e.g. the EDI burst files of the intervals without EFIELD mode, "null
    files" with only the times)."""

    try:
        out = reader(file_content, cdf_name, tint)
    except KeyError:
        if cdf_name in load(file_content):
            raise

        out = None

    return out


def _read_variables(file_content, cdf_name, tint: list | None = None):
    r"""Dataset of the variables of a file matching the pattern `cdf_name`
    (e.g. ``mms1_edi_*``): its data variables, their errors (``*_delta*``)
    and the energies of the GDUs (``*_energy_gdu*``), named without the
    spacecraft, instrument, rate and level, or None if the file has no data
    variable (a null file). The time dimension is ``time`` if the data
    variables have a single time base, else it is named after the base of
    each variable (e.g. ``time_90`` for ``Epoch_90``), as that of the
    energies when theirs differs (``time_timetag``)."""

    file = load(file_content)
    names = [
        name
        for name in file
        if fnmatch.fnmatchcase(name, cdf_name)
        and (_var_type(file[name]) == "data" or "_delta" in name)
    ]

    if not names:
        return None

    names += [
        name
        for name in file
        if fnmatch.fnmatchcase(name, cdf_name) and "_energy_gdu" in name
    ]
    bases = {name: file[name].attributes["DEPEND_0"][0] for name in names}
    data_bases = {bases[name] for name in names if "_energy_gdu" not in name}
    prefix = cdf_name.rstrip("*")
    out = {}

    for name in names:
        data = get_ts(file_content, name, tint)

        if len(data_bases) > 1 or bases[name] not in data_bases:
            base = bases[name].split("_", 1)[-1].lower()
            data = data.rename({data.dims[0]: f"time_{base}"})

        out["_".join(name.removeprefix(prefix).split("_")[:-2])] = data

    return Dataset(out)


def _var_type(variable) -> str:
    r"""VAR_TYPE of a CDF variable, empty without it."""

    attrs = variable.attributes

    if "VAR_TYPE" in attrs:
        return str(attrs["VAR_TYPE"][0])

    return ""


def _append_variables(out, data):
    r"""Datasets of `_read_variables` of two files appended in time, variable
    by variable."""

    if out is None:
        return data

    if set(out) != set(data):
        raise ValueError(
            f"The variables change between files ({sorted(out)}, {sorted(data)}): "
            "split the time interval"
        )

    return Dataset({key: ts_append(out[key], data[key]) for key in data})


def _listing_key(source: str, tint: list, mms_id: str, var: Mapping) -> dict:
    r"""Key of the listing of the files of `var` in `tint` in the cache: the
    source and its settings (bucket, SDC rights and user), the spacecraft, the
    instrument, rate, level and data type, and the time interval."""

    config = _read_config()

    if source == "aws":
        where = config["aws"]
    else:
        where = config["sdc"]

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
    cache (see :class:`pyrfu.mms._internal.cache.RemoteFile`)."""

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


def _resolve_source(source: str | None = "default") -> str:
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
    var: Mapping[str, str],
    data_path: str = "",
):
    # Time interval in the forms of time_clip, as datetime64; the cache key
    # of the listing is then the same for all the forms
    tint = _time_interval(tint)

    # Paths, cache files (RemoteFile), URLs or S3 objects, by source
    file_names: list[Any]
    headers: dict[str, str]

    if source == "local":
        file_names = list_files(tint, mms_id, var, data_path)
        sdc_session = None
        headers = {}
    elif source in ["sdc", "aws"] and _cache.enabled():
        # Files of the cache listing (the SDC login only if a file is read)
        items = _cache.listing(
            _listing_key(source, tint, mms_id, var),
            lambda: _remote_files(source, tint, mms_id, var),
        )
        file_names = [_cache.RemoteFile(**item) for item in items]
        sdc_session = None
        headers = {}
    elif source == "sdc":
        file_names = [file.get("url") for file in list_files_sdc(tint, mms_id, var)]
        sdc_session, headers, _ = _login_lasp()
    elif source == "aws":
        file_names = [file.get("s3_obj") for file in list_files_aws(tint, mms_id, var)]
        sdc_session = None
        headers = {}
    else:
        raise NotImplementedError(f"Resource {source} is not yet implemented!!")

    return file_names, sdc_session, headers


def _get_file_content_sources(
    source: str,
    file_name: str,
    sdc_session: Session | None = None,
    headers: dict | None = None,
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
            # The SDC session of _list_files_sources
            response = cast(Session, sdc_session).get(
                file_name, timeout=SDC_TIMEOUT, headers=headers
            )
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
    local path, of the key of an S3 object, of a file of the cache, or the
    file of an SDC download URL."""

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


@overload
def get_data(
    var_str: str,
    tint: list,
    mms_id: int | str,
    verbose: bool = ...,
    data_path: str = ...,
    source: str = ...,
) -> DataArray | Dataset: ...


@overload
def get_data(
    var_str: str,
    tint: list,
    mms_id: list[int | str] | tuple[int | str, ...] | range,
    verbose: bool = ...,
    data_path: str = ...,
    source: str = ...,
) -> list[DataArray | Dataset]: ...


def get_data(
    var_str: str,
    tint: list,
    mms_id: int | str | list[int | str] | tuple[int | str, ...] | range,
    verbose: bool = True,
    data_path: str = "",
    source: str = "",
) -> DataArray | Dataset | list[DataArray | Dataset]:
    r"""Load a variable `var_str`.

    AFG: b_bcs_afg_srvy_l2pre, b_dmpa_afg_srvy_l2pre, b_gse_afg_srvy_l2pre,
    b_gsm_afg_srvy_l2pre

    ASPOC: ebeam1_aspoc_srvy_l2, ebeam2_aspoc_srvy_l2, ionc1_aspoc_srvy_l2,
    ionc2_aspoc_srvy_l2, ionc_aspoc_srvy_l2, status_aspoc_srvy_l2

    DFG: b_bcs_dfg_srvy_l2pre, b_dmpa_dfg_srvy_l2pre, b_gse_dfg_srvy_l2pre,
    b_gsm_dfg_srvy_l2pre

    DSP: bpsd_dsp_fast_l2, bpsd_dsp_slow_l2, bpsdscm1_dsp_fast_l2, bpsdscm1_dsp_slow_l2,
    bpsdscm2_dsp_fast_l2, bpsdscm2_dsp_slow_l2, bpsdscm3_dsp_fast_l2,
    bpsdscm3_dsp_slow_l2, epsd_dsp_fast_l2, epsd_dsp_slow_l2, epsdx_dsp_fast_l2,
    epsdx_dsp_slow_l2, epsdy_dsp_fast_l2, epsdy_dsp_slow_l2, epsdz_dsp_fast_l2,
    epsdz_dsp_slow_l2, swde12_dsp_fast_l2, swde34_dsp_fast_l2

    EDI: counts-q0_edi_brst_l2, counts-q0_edi_srvy_l2, e_dsl_edi_brst_l2,
    e_dsl_edi_srvy_l2, e_gse_edi_brst_l2, e_gse_edi_srvy_l2, e_gsm_edi_brst_l2,
    e_gsm_edi_srvy_l2, errem_dsl_edi_brst_l2, errem_dsl_edi_srvy_l2,
    errem_gse_edi_brst_l2, errem_gse_edi_srvy_l2, errem_gsm_edi_brst_l2,
    errem_gsm_edi_srvy_l2, errep_dsl_edi_brst_l2, errep_dsl_edi_srvy_l2,
    errep_gse_edi_brst_l2, errep_gse_edi_srvy_l2, errep_gsm_edi_brst_l2,
    errep_gsm_edi_srvy_l2, errvdriftm_dsl_edi_brst_l2, errvdriftm_dsl_edi_srvy_l2,
    errvdriftm_gse_edi_brst_l2, errvdriftm_gse_edi_srvy_l2, errvdriftm_gsm_edi_brst_l2,
    errvdriftm_gsm_edi_srvy_l2, errvdriftp_dsl_edi_brst_l2, errvdriftp_dsl_edi_srvy_l2,
    errvdriftp_gse_edi_brst_l2, errvdriftp_gse_edi_srvy_l2, errvdriftp_gsm_edi_brst_l2,
    errvdriftp_gsm_edi_srvy_l2, flux-amb-alt-cc_edi_brst_l2,
    flux-amb-alt-cc_edi_srvy_l2, flux-amb-alt-oc_edi_brst_l2,
    flux-amb-alt-oc_edi_srvy_l2, flux-amb-alt-oob_edi_brst_l2,
    flux-amb-alt-oob_edi_srvy_l2, flux-amb-perp-ob_edi_brst_l2,
    flux-amb-perp-ob_edi_srvy_l2, flux-amb-pm2_edi_brst_l2, flux-amb-pm2_edi_srvy_l2,
    flux-amb_edi_brst_l2, flux-amb_edi_srvy_l2, trirchisq_edi_brst_l2,
    trirchisq_edi_srvy_l2, triweight_edi_brst_l2, triweight_edi_srvy_l2,
    vdrift_dsl_edi_brst_l2, vdrift_dsl_edi_srvy_l2, vdrift_gse_edi_brst_l2,
    vdrift_gse_edi_srvy_l2, vdrift_gsm_edi_brst_l2, vdrift_gsm_edi_srvy_l2

    EDP: bitmaske_edp_brst_l2, bitmaske_edp_fast_l2, bitmaske_edp_slow_l2,
    bitmaskv_edp_brst_l2, bitmaskv_edp_fast_l2, bitmaskv_edp_slow_l2,
    e2d_dsl_edp_brst_l2pre, e2d_dsl_edp_brst_ql, e2d_dsl_edp_fast_l2pre,
    e2d_dsl_edp_fast_ql, e_dsl_edp_brst_l2, e_dsl_edp_brst_l2pre, e_dsl_edp_brst_ql,
    e_dsl_edp_fast_l2, e_dsl_edp_fast_l2pre, e_dsl_edp_fast_ql, e_dsl_edp_slow_l2,
    e_dsl_edp_slow_l2pre, e_gse_edp_brst_l2, e_gse_edp_fast_l2, e_gse_edp_slow_l2,
    e_par_edp_brst_l2, e_par_edp_brst_l2pre, e_par_edp_fast_l2, e_par_edp_fast_l2pre,
    e_par_edp_slow_l2, e_par_edp_slow_l2pre, e_ssc_edp_brst_l2a, e_ssc_edp_fast_l2a,
    e_ssc_edp_slow_l2a, eres_dsl_edp_fast_l2, eres_dsl_edp_slow_l2, erre_edp_fast_l2,
    erre_edp_slow_l2, hfesp_edp_srvy_l2, hmfe_dsl_edp_brst_l2, hmfe_par_edp_brst_l2,
    phase_edp_fast_l2a, phase_edp_slow_l2a, qualitye_edp_brst_l2, qualitye_edp_fast_l2,
    qualitye_edp_slow_l2, qualityv_edp_brst_l2, qualityv_edp_fast_l2,
    qualityv_edp_slow_l2, sdev12_edp_fast_l2a, sdev12_edp_slow_l2a, sdev34_edp_fast_l2a,
    sdev34_edp_slow_l2a, v6_edp_brst_l2, v6_edp_fast_l2, v6_edp_slow_l2, v_edp_brst_l2,
    v_edp_fast_l2, v_edp_fast_sitl, v_edp_slow_l2, v_edp_slow_sitl, vpsp_edp_brst_l2,
    vpsp_edp_fast_l2, vpsp_edp_slow_l2

    FGM: b_bcs_fgm_brst_l2, b_bcs_fgm_srvy_l2, b_dmpa_fgm_brst_l2, b_dmpa_fgm_srvy_l2,
    b_gse_fgm_brst_l2, b_gse_fgm_srvy_l2, b_gsm_fgm_brst_l2, b_gsm_fgm_srvy_l2,
    etemp_fgm_brst_l2, etemp_fgm_srvy_l2, flag_fgm_brst_l2, flag_fgm_srvy_l2,
    hirange_fgm_brst_l2, hirange_fgm_srvy_l2, mode_fgm_brst_l2, mode_fgm_srvy_l2,
    r_gse_fgm_brst_l2, r_gse_fgm_srvy_l2, r_gsm_fgm_brst_l2, r_gsm_fgm_srvy_l2,
    stemp_fgm_brst_l2, stemp_fgm_srvy_l2

    FPI: avgf1countse_fpi_brst_l2, avgf1countse_fpi_fast_l2, avgf1countsi_fpi_brst_l2,
    avgf1countsi_fpi_fast_l2, bhate_dbcs_fpi_brst_l2, bhate_dbcs_fpi_fast_l2,
    bhati_dbcs_fpi_brst_l2, bhati_dbcs_fpi_fast_l2, defantie_fpi_brst_l2,
    defantie_fpi_fast_l2, defbgi_fpi_brst_l2, defbgi_fpi_fast_l2, defe_fpi_brst_l2,
    defe_fpi_fast_l2, defe_fpi_fast_ql, defi_fpi_brst_l2, defi_fpi_fast_l2,
    defmxe_fpi_brst_l2, defmxe_fpi_fast_l2, defmxi_fpi_brst_l2, defmxi_fpi_fast_l2,
    defmye_fpi_brst_l2, defmye_fpi_fast_l2, defmyi_fpi_brst_l2, defmyi_fpi_fast_l2,
    defmze_fpi_brst_l2, defmze_fpi_fast_l2, defmzi_fpi_brst_l2, defmzi_fpi_fast_l2,
    defpare_fpi_brst_l2, defpare_fpi_fast_l2, defperpe_fpi_brst_l2,
    defperpe_fpi_fast_l2, defpxe_fpi_brst_l2, defpxe_fpi_fast_l2, defpxi_fpi_brst_l2,
    defpxi_fpi_fast_l2, defpye_fpi_brst_l2, defpye_fpi_fast_l2, defpyi_fpi_brst_l2,
    defpyi_fpi_fast_l2, defpze_fpi_brst_l2, defpze_fpi_fast_l2, defpzi_fpi_brst_l2,
    defpzi_fpi_fast_l2, errne_fpi_brst_l2, errne_fpi_fast_l2, errni_fpi_brst_l2,
    errni_fpi_fast_l2, errpe_fpi_brst_l2, errpe_fpi_fast_l2, errpi_fpi_brst_l2,
    errpi_fpi_fast_l2, errqe_fpi_brst_l2, errqe_fpi_fast_l2, errqi_fpi_brst_l2,
    errqi_fpi_fast_l2, errte_fpi_brst_l2, errte_fpi_fast_l2, errti_fpi_brst_l2,
    errti_fpi_fast_l2, errve_fpi_brst_l2, errve_fpi_fast_l2, errvi_fpi_brst_l2,
    errvi_fpi_fast_l2, flage_fpi_brst_l2, flage_fpi_fast_l2, flagi_fpi_brst_l2,
    flagi_fpi_fast_l2, nbgi_fpi_brst_l2, nbgi_fpi_fast_l2, ne_fpi_brst_l2,
    ne_fpi_fast_l2, ne_fpi_fast_ql, nexthighe_fpi_brst_l2, nexthighe_fpi_fast_l2,
    nexthighi_fpi_brst_l2, nexthighi_fpi_fast_l2, nextlowe_fpi_brst_l2,
    nextlowe_fpi_fast_l2, nextlowi_fpi_brst_l2, nextlowi_fpi_fast_l2, ni_fpi_brst_l2,
    ni_fpi_fast_l2, ni_fpi_fast_ql, padhighene_fpi_brst_l2, padhighene_fpi_fast_l2,
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
    pderri_fpi_fast_l2, pdi_fpi_brst_l2, pdi_fpi_fast_l2, pdphotoe_fpi_brst_l2,
    pdphotoe_fpi_fast_l2, pdtoffe_fpi_brst_l2, pdtoffi_fpi_brst_l2, pe_dbcs_fpi_brst_l2,
    pe_dbcs_fpi_fast_l2, pe_dbcs_fpi_fast_ql, pe_gse_fpi_brst_l2, pe_gse_fpi_fast_l2,
    pe_gse_fpi_fast_ql, pi_dbcs_fpi_brst_l2, pi_dbcs_fpi_fast_l2, pi_dbcs_fpi_fast_ql,
    pi_gse_fpi_brst_l2, pi_gse_fpi_fast_l2, pi_gse_fpi_fast_ql, qe_dbcs_fpi_brst_l2,
    qe_dbcs_fpi_fast_l2, qe_gse_fpi_brst_l2, qe_gse_fpi_fast_l2, qi_dbcs_fpi_brst_l2,
    qi_dbcs_fpi_fast_l2, qi_gse_fpi_brst_l2, qi_gse_fpi_fast_l2, scpmeane_fpi_brst_l2,
    scpmeane_fpi_fast_l2, scpmeani_fpi_brst_l2, scpmeani_fpi_fast_l2,
    ste_dbcs_fpi_brst_l2, ste_dbcs_fpi_fast_l2, ste_gse_fpi_brst_l2,
    ste_gse_fpi_fast_l2, sti_dbcs_fpi_brst_l2, sti_dbcs_fpi_fast_l2,
    sti_gse_fpi_brst_l2, sti_gse_fpi_fast_l2, te_dbcs_fpi_brst_l2, te_dbcs_fpi_fast_l2,
    te_dbcs_fpi_fast_ql, te_gse_fpi_brst_l2, te_gse_fpi_fast_l2, te_gse_fpi_fast_ql,
    ti_dbcs_fpi_brst_l2, ti_dbcs_fpi_fast_l2, ti_gse_fpi_brst_l2, ti_gse_fpi_fast_l2,
    tparae_fpi_brst_l2, tparae_fpi_fast_l2, tparai_fpi_brst_l2, tparai_fpi_fast_l2,
    tperpe_fpi_brst_l2, tperpe_fpi_fast_l2, tperpi_fpi_brst_l2, tperpi_fpi_fast_l2,
    ve_dbcs_fpi_brst_l2, ve_dbcs_fpi_fast_l2, ve_dbcs_fpi_fast_ql, ve_gse_fpi_brst_l2,
    ve_gse_fpi_fast_l2, ve_gse_fpi_fast_ql, vi_dbcs_fpi_brst_l2, vi_dbcs_fpi_fast_l2,
    vi_dbcs_fpi_fast_ql, vi_gse_fpi_brst_l2, vi_gse_fpi_fast_l2, vi_gse_fpi_fast_ql,
    xforme_gse_fpi_brst_l2, xforme_gse_fpi_fast_l2, xformi_gse_fpi_brst_l2,
    xformi_gse_fpi_fast_l2

    FSM: b_gse_fsm_brst_l3, r_gse_fsm_brst_l3

    HPCA: azimuth_hpca_brst_l2, azimuth_hpca_srvy_l2, b_dmpa_hpca_brst_l2,
    b_dmpa_hpca_srvy_l2, b_gsm_hpca_brst_l2, b_gsm_hpca_srvy_l2,
    bsweep_dmpa_hpca_brst_l2, bsweep_dmpa_hpca_srvy_l2, bsweep_gsm_hpca_brst_l2,
    bsweep_gsm_hpca_srvy_l2, dpfheplus_hpca_brst_l2, dpfheplus_hpca_srvy_l2,
    dpfheplusplus_hpca_brst_l2, dpfheplusplus_hpca_srvy_l2, dpfhplus_hpca_brst_l2,
    dpfhplus_hpca_srvy_l2, dpfoplus_hpca_brst_l2, dpfoplus_hpca_srvy_l2,
    flagheplus_hpca_brst_l2, flagheplus_hpca_srvy_l2, flagheplusplus_hpca_brst_l2,
    flagheplusplus_hpca_srvy_l2, flaghplus_hpca_brst_l2, flaghplus_hpca_srvy_l2,
    flagoplus_hpca_brst_l2, flagoplus_hpca_srvy_l2, mcpvmon_hpca_brst_l2,
    mcpvmon_hpca_srvy_l2, nheplus_hpca_brst_l2, nheplus_hpca_srvy_l2,
    nheplusplus_hpca_brst_l2, nheplusplus_hpca_srvy_l2, nhplus_hpca_brst_l2,
    nhplus_hpca_srvy_l2, noplus_hpca_brst_l2, noplus_hpca_srvy_l2,
    pheplus_dbcs_hpca_brst_l2, pheplus_dbcs_hpca_srvy_l2, pheplusplus_dbcs_hpca_brst_l2,
    pheplusplus_dbcs_hpca_srvy_l2, phplus_dbcs_hpca_brst_l2, phplus_dbcs_hpca_srvy_l2,
    poplus_dbcs_hpca_brst_l2, poplus_dbcs_hpca_srvy_l2, saz_hpca_brst_l2,
    saz_hpca_srvy_l2, spinnum_hpca_brst_l2, spinnum_hpca_srvy_l2,
    theplus_dbcs_hpca_brst_l2, theplus_dbcs_hpca_srvy_l2, theplusplus_dbcs_hpca_brst_l2,
    theplusplus_dbcs_hpca_srvy_l2, thplus_dbcs_hpca_brst_l2, thplus_dbcs_hpca_srvy_l2,
    tofcounts_hpca_srvy_l2, tofvmon_hpca_brst_l2, tofvmon_hpca_srvy_l2,
    toplus_dbcs_hpca_brst_l2, toplus_dbcs_hpca_srvy_l2, tparaheplus_hpca_brst_l2,
    tparaheplus_hpca_srvy_l2, tparaheplusplus_hpca_brst_l2,
    tparaheplusplus_hpca_srvy_l2, tparahplus_hpca_brst_l2, tparahplus_hpca_srvy_l2,
    tparaoplus_hpca_brst_l2, tparaoplus_hpca_srvy_l2, tperpheplus_hpca_brst_l2,
    tperpheplus_hpca_srvy_l2, tperpheplusplus_hpca_brst_l2,
    tperpheplusplus_hpca_srvy_l2, tperphplus_hpca_brst_l2, tperphplus_hpca_srvy_l2,
    tperpoplus_hpca_brst_l2, tperpoplus_hpca_srvy_l2, tsheplus_hpca_brst_l2,
    tsheplus_hpca_srvy_l2, tsheplusplus_hpca_brst_l2, tsheplusplus_hpca_srvy_l2,
    tshplus_hpca_brst_l2, tshplus_hpca_srvy_l2, tsoplus_hpca_brst_l2,
    tsoplus_hpca_srvy_l2, vheplus_dbcs_hpca_brst_l2, vheplus_dbcs_hpca_srvy_l2,
    vheplus_gsm_hpca_brst_l2, vheplus_gsm_hpca_srvy_l2, vheplusplus_dbcs_hpca_brst_l2,
    vheplusplus_dbcs_hpca_srvy_l2, vheplusplus_gsm_hpca_brst_l2,
    vheplusplus_gsm_hpca_srvy_l2, vhplus_dbcs_hpca_brst_l2, vhplus_dbcs_hpca_srvy_l2,
    vhplus_gsm_hpca_brst_l2, vhplus_gsm_hpca_srvy_l2, voplus_dbcs_hpca_brst_l2,
    voplus_dbcs_hpca_srvy_l2, voplus_gsm_hpca_brst_l2, voplus_gsm_hpca_srvy_l2,
    vparaheplus_dbcs_hpca_brst_l2, vparaheplus_dbcs_hpca_srvy_l2,
    vparaheplusplus_dbcs_hpca_brst_l2, vparaheplusplus_dbcs_hpca_srvy_l2,
    vparahplus_dbcs_hpca_brst_l2, vparahplus_dbcs_hpca_srvy_l2,
    vparaoplus_dbcs_hpca_brst_l2, vparaoplus_dbcs_hpca_srvy_l2,
    vperpheplus_dbcs_hpca_brst_l2, vperpheplus_dbcs_hpca_srvy_l2,
    vperpheplusplus_dbcs_hpca_brst_l2, vperpheplusplus_dbcs_hpca_srvy_l2,
    vperphplus_dbcs_hpca_brst_l2, vperphplus_dbcs_hpca_srvy_l2,
    vperpoplus_dbcs_hpca_brst_l2, vperpoplus_dbcs_hpca_srvy_l2

    MEC: bfnt89d_gsm_mec_brst_l2, bfnt89d_gsm_mec_srvy_l2, bfnt89q_gsm_mec_brst_l2,
    bfnt89q_gsm_mec_srvy_l2, bfnts04d_gsm_mec_brst_l2, bfnts04d_gsm_mec_srvy_l2,
    bfst89d_gsm_mec_brst_l2, bfst89d_gsm_mec_srvy_l2, bfst89q_gsm_mec_brst_l2,
    bfst89q_gsm_mec_srvy_l2, bfsts04d_gsm_mec_brst_l2, bfsts04d_gsm_mec_srvy_l2,
    bmint89d_gsm_mec_brst_l2, bmint89d_gsm_mec_srvy_l2, bmint89q_gsm_mec_brst_l2,
    bmint89q_gsm_mec_srvy_l2, bmints04d_gsm_mec_brst_l2, bmints04d_gsm_mec_srvy_l2,
    bmodt89d_gsm_mec_brst_l2, bmodt89d_gsm_mec_srvy_l2, bmodt89q_gsm_mec_brst_l2,
    bmodt89q_gsm_mec_srvy_l2, bmodts04d_gsm_mec_brst_l2, bmodts04d_gsm_mec_srvy_l2,
    dst_mec_brst_l2, dst_mec_srvy_l2, eclipseearth_mec_brst_l2,
    eclipseearth_mec_srvy_l2, eclipsemoon_mec_brst_l2, eclipsemoon_mec_srvy_l2,
    fltypet89d_mec_brst_l2, fltypet89d_mec_srvy_l2, fltypet89q_mec_brst_l2,
    fltypet89q_mec_srvy_l2, fltypets04d_mec_brst_l2, fltypets04d_mec_srvy_l2,
    geodh_mec_brst_l2, geodh_mec_srvy_l2, geodlat_mec_brst_l2, geodlat_mec_srvy_l2,
    geodlon_mec_brst_l2, geodlon_mec_srvy_l2, gmst_mec_brst_l2, gmst_mec_srvy_l2,
    kp_mec_brst_l2, kp_mec_srvy_l2, latlonfnt89d_mec_brst_l2, latlonfnt89d_mec_srvy_l2,
    latlonfnt89q_mec_brst_l2, latlonfnt89q_mec_srvy_l2, latlonfnts04d_mec_brst_l2,
    latlonfnts04d_mec_srvy_l2, latlonfst89d_mec_brst_l2, latlonfst89d_mec_srvy_l2,
    latlonfst89q_mec_brst_l2, latlonfst89q_mec_srvy_l2, latlonfsts04d_mec_brst_l2,
    latlonfsts04d_mec_srvy_l2, ldip_mec_brst_l2, ldip_mec_srvy_l2,
    losscnt89d_mec_brst_l2, losscnt89d_mec_srvy_l2, losscnt89q_mec_brst_l2,
    losscnt89q_mec_srvy_l2, losscnts04d_mec_brst_l2, losscnts04d_mec_srvy_l2,
    losscst89d_mec_brst_l2, losscst89d_mec_srvy_l2, losscst89q_mec_brst_l2,
    losscst89q_mec_srvy_l2, losscsts04d_mec_brst_l2, losscsts04d_mec_srvy_l2,
    lphase_mec_brst_l2, lphase_mec_srvy_l2, lvec_mec_brst_l2, lvec_mec_srvy_l2,
    mlat_mec_brst_l2, mlat_mec_srvy_l2, mlt_mec_brst_l2, mlt_mec_srvy_l2,
    pphase_mec_brst_l2, pphase_mec_srvy_l2, pvec_mec_brst_l2, pvec_mec_srvy_l2,
    quat_bcs_mec_brst_l2, quat_bcs_mec_srvy_l2, quat_dbcs_mec_brst_l2,
    quat_dbcs_mec_srvy_l2, quat_dmpa_mec_brst_l2, quat_dmpa_mec_srvy_l2,
    quat_dsl_mec_brst_l2, quat_dsl_mec_srvy_l2, quat_geo_mec_brst_l2,
    quat_geo_mec_srvy_l2, quat_gse2000_mec_brst_l2, quat_gse2000_mec_srvy_l2,
    quat_gse_mec_brst_l2, quat_gse_mec_srvy_l2, quat_gsm_mec_brst_l2,
    quat_gsm_mec_srvy_l2, quat_sm_mec_brst_l2, quat_sm_mec_srvy_l2,
    quat_smpa_mec_brst_l2, quat_smpa_mec_srvy_l2, quat_ssl_mec_brst_l2,
    quat_ssl_mec_srvy_l2, r_eci_mec_brst_l2, r_eci_mec_srvy_l2, r_geo_mec_brst_l2,
    r_geo_mec_srvy_l2, r_gse2000_mec_brst_l2, r_gse2000_mec_srvy_l2, r_gse_mec_brst_l2,
    r_gse_mec_srvy_l2, r_gsm_mec_brst_l2, r_gsm_mec_srvy_l2, r_sm_mec_brst_l2,
    r_sm_mec_srvy_l2, rfnt89d_gsm_mec_brst_l2, rfnt89d_gsm_mec_srvy_l2,
    rfnt89q_gsm_mec_brst_l2, rfnt89q_gsm_mec_srvy_l2, rfnts04d_gsm_mec_brst_l2,
    rfnts04d_gsm_mec_srvy_l2, rfst89d_gsm_mec_brst_l2, rfst89d_gsm_mec_srvy_l2,
    rfst89q_gsm_mec_brst_l2, rfst89q_gsm_mec_srvy_l2, rfsts04d_gsm_mec_brst_l2,
    rfsts04d_gsm_mec_srvy_l2, rmint89d_gsm_mec_brst_l2, rmint89d_gsm_mec_srvy_l2,
    rmint89q_gsm_mec_brst_l2, rmint89q_gsm_mec_srvy_l2, rmints04d_gsm_mec_brst_l2,
    rmints04d_gsm_mec_srvy_l2, rmoon_eci_mec_brst_l2, rmoon_eci_mec_srvy_l2,
    rsun_eci_mec_brst_l2, rsun_eci_mec_srvy_l2, tilt_mec_brst_l2, tilt_mec_srvy_l2,
    vel_eci_mec_brst_l2, vel_eci_mec_srvy_l2, vel_geo_mec_brst_l2, vel_geo_mec_srvy_l2,
    vel_gse2000_mec_brst_l2, vel_gse2000_mec_srvy_l2, vel_gse_mec_brst_l2,
    vel_gse_mec_srvy_l2, vel_gsm_mec_brst_l2, vel_gsm_mec_srvy_l2, vel_sm_mec_brst_l2,
    vel_sm_mec_srvy_l2, zphase_mec_brst_l2, zphase_mec_srvy_l2, zvec_mec_brst_l2,
    zvec_mec_srvy_l2

    SCM: b_gse_scm_brst_l2, b_gse_scm_srvy_l2, bschb_gse_scm_brst_l2

    Parameters
    ----------
    var_str : str
        Key of the target variable (use mms.get_data() to see keys.).
    tint : array_like or xarray.DataArray
        Time interval: the start and stop times as ISO 8601 str, datetime64
        or datetime.datetime, or a time series (its first and last times), as
        `pyrfu.pyrf.time_clip`.
    mms_id : int or str or sequence of int or str
        Index of the target spacecraft, or of several spacecraft (e.g.
        ``[1, 2, 3, 4]``), whose time series are returned as a list, in the
        same order.
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
    out : DataArray or Dataset or list
        Time series of the target variable of measured by the target
        spacecraft over the selected time interval; for several spacecraft,
        the list of their time series, each on its own times (no
        resampling). A spectrum whose axis changes with time (e.g. the FPI
        burst energy spectra of the early mission, with two alternating
        energy tables) is a Dataset as `pyrfu.pyrf.ts_spectr`: the data
        (time, idx0) and the axis (time, idx0).

    Raises
    ------
    FileNotFoundError
        If no file of the variable covers the time interval.
    KeyError
        If none of the files has the variable.
    ValueError
        If `source` is not "local", "sdc", "aws" or "default".

    See also
    --------
    pyrfu.mms.get_ts : Read time series.
    pyrfu.mms.get_dist : Read velocity distribution function.

    Notes
    -----
    The FPI distributions (``pd*``, ``pderr*``, ``pdphotoe``, the modelled
    photoelectron distribution of the DES ``distaux`` files, and ``pdtoff*``,
    the time offsets of the bins from the time of the distribution, us, burst
    files of version 3.4 and later) are skymaps, read by
    `pyrfu.mms.get_dist`. The other variables of the distribution files are
    time series, e.g. ``avgf1counts*``, the average one-count level of each
    energy (s^3/cm^6; files of version 3.4 and later): for DIS it is within
    2 % of the mean one-count level of the bins from the L2 distribution and
    its error (``vdf_one_count``), for DES 4/3 of it (on 2017-07-11; a
    question to the FPI team). The directional
    spectra ``def{px,mx,py,my,pz,mz}*`` are those near the +X, -X, +Y, -Y,
    +Z and -Z axes of DBCS, ``bhat*_dbcs`` the unit vector of B and
    ``xform*_gse`` the DBCS to GSE matrix used by FPI for the moments (CMAD
    Table 4-8).

    The EDI uncertainties are the lower and upper errors of the E field and
    drift velocity (``errem``, ``errep``, ``errvdriftm``, ``errvdriftp``),
    with the weight of the TRI method in the result (``triweight``, %) and
    its reduced chi-squared (``trirchisq``).

    The HPCA flux and phase-space density are read with their energies and
    look directions by `pyrfu.mms.get_hpca_dist` and `pyrfu.mms.hpca_skymap`.

    The keys of the levels l2pre, l2a, ql and sitl (AFG, DFG, EDP and FPI)
    read files of the MMS team, at the SDC with a team login (see
    `pyrfu.mms.db_init`): they are neither in the public SDC files nor in
    the public copy of the archive on AWS.

    The keys of the EDI ambient and quality zero datasets (``flux-amb*_edi_*``,
    ``counts-q0_edi_*``) give a Dataset of all the fluxes (or counts) of the
    file, their errors (``*_delta*``), the trajectories of the electrons and
    the energies of the GDUs (``energy_gdu1``, ``energy_gdu2``, eV, on the time
    base ``time_timetag`` in the ambient files: 500 eV in 2017 and 2019, 250 eV
    in 2022), e.g. ``flux1_0``, ``flux1_0_delta``, ``traj1_gse_0``: the fluxes
    at 0 and 180 degrees pitch angle of channel 1 (and of channels 2 to 4 in
    burst amb-pm2), at 90 degrees of each GDU in the amb-alt and amb-perp modes
    (CMAD pp. 344-357). In the files with several time bases (amb-alt: 0 and
    180 degrees, 90 degrees; q0: each GDU) the time dimension of each variable
    is named after its base, e.g. ``time_0_180`` and ``time_90``. The q0
    trajectories of GDU2 are in degrees, as those of GDU1, although their UNITS
    say rad.

    The MEC variables of a field model are in the files of each model (T89
    with Kp, T89 quiet, TS04 dynamic): their keys end with the model, e.g.
    ``bmodt89d_gsm_mec_srvy_l2``, ``bmodt89q_gsm_mec_srvy_l2`` and
    ``bmodts04d_gsm_mec_srvy_l2`` (the model field at the spacecraft; ``bfn``,
    ``bfs``, ``bmin`` at the northern and southern footpoints at 100 km and at
    the minimum of B, ``rfn``, ``rfs``, ``rmin`` their positions,
    ``latlonfn``, ``latlonfs`` the geodetic footpoints, ``losscn``,
    ``losscs`` the loss cone angles, ``fltype`` the field line type, CMAD
    Table 3-5). The keys of the other MEC variables read the T89 (Kp) files,
    which have them all.

    The files without the variable are skipped, with a warning: e.g. the EDI
    burst files of the intervals without EFIELD mode, "null files" with only
    the times (the 67982 MMS1 burst efield files of version 0.0.0, and files
    of later versions, e.g. on 2015-09-14).

    The HPCA velocities in GSM (``v*_gsm_hpca_*``) are those of the files,
    the DBCS velocities rotated from GSE to GSM as if DBCS were GSE: the
    rotation by the tilt of the spin axis (2 to 4 degrees) is missing. For
    the velocities in GSM, rotate the DBCS velocities (``v*_dbcs_hpca_*``)
    with `pyrfu.mms.dsl2gsm` and the definitive attitude (``defatt``,
    `pyrfu.mms.load_ancillary`). The HPCA magnetic field ``B_GSE_spin_avg``
    of the files is in DMPA (``b_dmpa_hpca_*``), as ``B_GSE_sweep_avg``
    (``bsweep_dmpa_hpca_*``), whose four columns (x, y, z and the magnitude)
    are all read: the files do not label them as a vector.

    The DSP solitary wave counts are those of E12 (``swde12_dsp_fast_l2``)
    on the four spacecraft in 2017, and of E34 (``swde34_dsp_fast_l2``) on
    MMS2 and MMS3 in 2023.

    .. deprecated:: 2.6.0
        The ``v*_gsm_hpca_*`` keys will be removed in pyrfu 3.0.0.

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint_brst = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]

    Index of MMS spacecraft

    >>> ic = 1

    Load magnetic field from FGM

    >>> b_xyz = mms.get_data("b_gse_fgm_brst_l2", tint_brst, ic)

    Load the magnetic field of the four spacecraft

    >>> b_mms = mms.get_data("b_gse_fgm_brst_l2", tint_brst, [1, 2, 3, 4])

    """

    # The HPCA velocities in GSM (deprecated since 2.6.0)
    _warn_deprecated_key(var_str)

    # Several spacecraft (a sequence of indices): the list of their time
    # series, in the same order
    if isinstance(mms_id, (str, int, Integral)):
        out: Any = _get_data(var_str, tint, str(mms_id), verbose, data_path, source)
    else:
        out = [
            _get_data(var_str, tint, str(i), verbose, data_path, source) for i in mms_id
        ]

    return out


def _get_data(
    var_str: str,
    tint: list,
    mms_id: str,
    verbose: bool,
    data_path: str,
    source: str,
) -> DataArray | Dataset:
    r"""Time series of the variable `var_str` of the spacecraft `mms_id`
    (str), see `get_data`."""

    var, cdf_name = _var_and_cdf_name(var_str, mms_id)

    # The source in any case, or the default of the MMS configuration if empty
    # or "default"; ValueError for an unknown one, as db_get_ts,
    # db_get_variable, get_eis_allt and load_ancillary
    source = _resolve_source(source)

    file_names, sdc_session, headers = _list_files_sources(
        source, tint, mms_id, var, data_path
    )

    out = None

    if not file_names:
        raise FileNotFoundError(f"No files found for {var_str} in {source}")

    if verbose:
        logger.info("Loading %s...", cdf_name)

    # The skymaps of the FPI distribution files with get_dist, all the
    # variables of a file (a pattern) as a Dataset, the other variables with
    # get_ts
    reader: Callable[..., Any]
    append: Callable[..., Any]

    if "-dist" in var["dtype"] and _is_skymap(cdf_name):
        reader, append = get_dist, dist_append
    elif cdf_name.endswith("*"):
        reader, append = _read_variables, _append_variables
    else:
        reader, append = get_ts, ts_append

    for file_name in file_names:
        file_content = _get_file_content_sources(
            source, file_name, sdc_session, headers
        )
        data = _read(reader, file_content, cdf_name, tint)

        if data is None:
            logger.warning(
                "%s is not in %s: file skipped", cdf_name, _file_label(file_name)
            )
        else:
            out = append(out, data)

    if out is None:
        raise KeyError(f"{cdf_name} is in none of the files of {var_str}")

    return cast(DataArray | Dataset, out)
