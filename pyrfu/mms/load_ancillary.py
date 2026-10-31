#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import bisect
import io
import json
import logging
import os

# 3rd party imports
import numpy as np
import pandas as pd
import requests

# Local imports
from pyrfu.mms.db_get_ts import _resolve_source
from pyrfu.mms.list_files_ancillary import _sc_name, list_files_ancillary
from pyrfu.mms.list_files_ancillary_sdc import list_files_ancillary_sdc
from pyrfu.mms.list_files_sdc import SDC_TIMEOUT, _login_lasp
from pyrfu.pyrf.extend_tint import extend_tint
from pyrfu.pyrf.iso86012datetime import iso86012datetime
from pyrfu.pyrf.iso86012datetime64 import iso86012datetime64

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logger = logging.getLogger(__name__)

# Margin on the end of the SDC files, as list_files_ancillary for the local
# files: the file names give dates only, and a file can cover up to a day
# after its end date (MMS1 DEFATT 2015361_2015362: 2015-361T15:56 to
# 362T15:54; 2017: 00:00 to 00:05 the day after)
_END_MARGIN = np.timedelta64(1, "D")

# Units of the columns (DEFEPH as its header and IDL SPEDAS; DEFATT from the
# values: wZ of about 18 deg/s for the spin period of 20 s; the DEFATT quality
# flag is a string, without units)
_ANGLES = ["w_phase", "z_ra", "z_dec", "z_phase", "l_ra", "l_dec", "l_phase"]
_ANGLES += ["p_ra", "p_dec", "p_phase", "nut"]
_UNITS = {
    "defatt": {
        "elapsed_sec": "s",
        **dict.fromkeys(["q1", "q2", "q3", "qc"], ""),
        **dict.fromkeys(["wx", "wy", "wz"], "deg/s"),
        **dict.fromkeys(_ANGLES, "deg"),
    },
    "defeph": {
        "epoch_mms_tai": "d",
        **dict.fromkeys(["x", "y", "z"], "km"),
        **dict.fromkeys(["vx", "vy", "vz"], "km/s"),
    },
    "defq": {"epoch_mms_tai": "d", "q": "", "scale": "km", "maneuver_flag": ""},
}


def _list_files_sdc(tint, mms_id, product):
    r"""Ancillary files of the SDC covering tint, the latest version of each
    period, in time order (as list_files_ancillary for the local files)."""
    tint_0, tint_1 = iso86012datetime64(np.array(tint))
    files = list_files_ancillary_sdc(extend_tint(tint, [-86400, 0]), mms_id, product)

    def covers(file):
        start, end = iso86012datetime64(
            np.array([file["start_date"], file["end_date"]])
        )
        return start <= tint_1 and end + _END_MARGIN >= tint_0

    latest = {}

    for file in filter(covers, files):
        period, version = file["file_name"].rsplit(".", 1)

        if period not in latest or version > latest[period][0]:
            latest[period] = (version, file)

    return [latest[period][1] for period in sorted(latest)]


def _read_file(file, product_info, tint):
    r"""Rows of an ancillary file (path or file-like) in tint, with the last of
    the rows repeating a time (as mms_load_ancillary)."""
    # All the columns of the product: the rows may leave the last ones empty
    # (the maneuver flag of DEFQ) or have no data ("No data exists")
    rows = pd.read_csv(
        file,
        sep=r"\s+",
        header=None,
        skiprows=product_info["header"],
        names=range(len(product_info["columns_names"])),
        index_col=False,
    )

    # Convert time, and remove the rows that are not data, such as the
    # DATA_STOP footer of DEFATT files (DEFEPH files have no footer)
    fmt = product_info["time_format"]
    rows[0] = pd.to_datetime(rows[0], format=fmt, errors="coerce")
    rows = rows[rows[0].notna()].reset_index(drop=True)
    start_idx = bisect.bisect_left(rows[0][:], tint[0])
    end_idx = bisect.bisect_left(rows[0][:], tint[1])
    rows.columns = product_info["columns_names"]

    return rows[start_idx:end_idx].drop_duplicates(subset="time", keep="last")


def load_ancillary(
    product,
    tint,
    mms_id,
    verbose: bool = True,
    data_path: str = "",
    source: str = "default",
):
    r"""Loads ancillary data.

    Parameters
    ----------
    product : {"defatt", "defeph", "defq"}
        Ancillary type. The predicted products (predatt, predeph, predq) are
        not supported.
    tint : list of str
        Time interval
    mms_id : str or int
        Spacecraft index (not used for "defq", which covers the four
        spacecraft).
    verbose : bool, Optional
        Set to True to follow the loading. Default is True
    data_path : str, Optional
        Path of MMS data, for the local files. If None use
        `pyrfu.mms.mms_config.py`
    source : {"default", "local", "sdc", "aws"}, Optional
        Resource to fetch the data from. Default uses default in
        `pyrfu/mms/config.json`. The ancillary files are not on AWS (its MMS
        archive is the CDAWeb copy of the CDF files): "aws" reads them from the
        SDC.

    Returns
    -------
    out : xarray.Dataset
        Time series of the ancillary data, with strictly increasing times, and
        the units of the columns in their UNITS attribute. "defeph": the
        position (km) and velocity (km/s), in GEI J2000 as the file headers.
        "defatt": the quaternion (""), the angular velocity (deg/s),
        the right ascensions, declinations, phases and nutation angle (deg),
        and the quality flag (a string, no UNITS). "defq", every 30 s: the
        tetrahedron quality factor q ("", 1 for a regular tetrahedron), the
        scale of the file (km), and the maneuver flag ("", NaN when not set).
        epoch_mms_tai is in days and elapsed_sec in seconds since
        1958-01-01 (TAI).

    Raises
    ------
    ValueError
        If the product is not supported.
    FileNotFoundError
        If no file of the product covers the time interval.
    requests.HTTPError
        If a file cannot be read from the SDC.

    Notes
    -----
    DEFATT files repeat some times by design, and consecutive files overlap
    (by about 5 minutes, with phases different by up to 0.007 deg). As
    irfu-matlab, the last of the repeated rows of a file is kept
    (mms_load_ancillary), and in an overlap the rows of the earlier file
    (mms_db).

    From the SDC, the files are read in memory, the whole files covering the
    time interval (DEFATT: one file of about 40 MB a day). To load them many
    times, download them once with `download_ancillary` and use
    ``source="local"``.

    """

    # Read length of header and columns names from .json file
    # Root path
    pkg_path = os.path.dirname(os.path.abspath(__file__))

    with open(os.sep.join([pkg_path, "ancillary.json"]), "r", encoding="utf-8") as file:
        anc_dict = json.load(file)

    if product not in anc_dict:
        raise ValueError(
            f"Ancillary product {product!r} is not supported, use one of "
            f"{', '.join(anc_dict)}"
        )

    # Resource: the ancillary files are not on AWS, read them from the SDC
    resource = _resolve_source(source)

    if resource == "aws":
        logger.info("No ancillary files on AWS: %s read from the SDC", product)
        resource = "sdc"

    # Files covering the time interval (and the SDC session to read them)
    sdc_session, headers = None, None

    if resource == "local":
        tint_long = extend_tint(tint, [-86400, 86400])
        files_names = list_files_ancillary(tint_long, mms_id, product, data_path)
    else:
        files_names = _list_files_sdc(tint, mms_id, product)
        sdc_session, headers, _ = _login_lasp()

    if not files_names:
        raise FileNotFoundError(
            f"No {product} file found for {_sc_name(product, mms_id).upper()} in "
            f"{tint}"
        )

    # Convert time interval to datetime
    tint = iso86012datetime(tint)

    if verbose:
        logger.info("Loading ancillary %s files...", product)

    data_frame_dict = {}

    for i, file in enumerate(files_names):
        if resource == "sdc":
            try:
                response = sdc_session.get(
                    file["url"], timeout=SDC_TIMEOUT, headers=headers
                )
                response.raise_for_status()
            except requests.RequestException:
                logger.error("Error retrieving file from %s", file["url"])
                raise

            file = io.BytesIO(response.content)

        data_frame_dict[i] = _read_file(file, anc_dict[product], tint)

    data_frame = pd.concat(list(data_frame_dict.values()), ignore_index=True)

    # Consecutive files overlap: the rows of the earlier file, as mms_db (the
    # files are in time order, and the sort is stable)
    data_frame = data_frame.sort_values(by="time", kind="stable")
    data_frame = data_frame.drop_duplicates(subset="time", keep="first")
    data_frame = data_frame.set_index(["time"])
    dataset_out = data_frame.to_xarray()

    dataset_out = dataset_out.assign_coords(time=dataset_out.time.astype("<M8[ns]"))

    for name, units in _UNITS[product].items():
        dataset_out[name].attrs["UNITS"] = units

    return dataset_out
