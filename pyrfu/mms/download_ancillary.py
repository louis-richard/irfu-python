#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import os
from typing import Literal, Optional, Union

# Local imports
from pyrfu.mms.db_init import _read_config
from pyrfu.mms.download_data import _download_file
from pyrfu.mms.list_files_ancillary import _sc_name
from pyrfu.mms.list_files_ancillary_sdc import list_files_ancillary_sdc
from pyrfu.mms.list_files_sdc import _login_lasp

# 3rd party imports


__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"

logger = logging.getLogger(__name__)


LASP_PUBL = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1/"
LASP_SITL = "https://lasp.colorado.edu/mms/sdc/sitl/files/api/v1/"


def _make_path_local(
    file: dict,
    product: Literal["predatt", "predeph", "predq", "defatt", "defeph", "defq"],
    mms_id: Union[str, int],
    data_path: Optional[str] = "",
) -> str:
    r"""Construct path of the data file using the standard convention.

    Parameters
    ----------
    file : dict
        File information.
    product : {"predatt", "predeph", "predq", "defatt", "defeph", "defq"}
        Ancillary type.
    mms_id : str or int
        Spacecraft index.
    data_path : str, Optional
        Path of MMS data. Default uses the local path of the MMS configuration
        (`pyrfu.mms.MMS_CFG_PATH`, set with `mms.db_init`).

    Returns
    -------
    str
        Full path of the data file.

    Raises
    ------
    FileNotFoundError
        If the local data directory doesn't exist.

    """
    data_path = os.path.normpath(data_path or _read_config()["local"])

    if not os.path.exists(data_path):
        raise FileNotFoundError("local data directory doesn't exist!")

    path_list = [
        data_path,
        "ancillary",
        _sc_name(product, mms_id),
        product,
    ]

    return os.path.join(*path_list, file["file_name"])


def download_ancillary(
    product: Literal["predatt", "predeph", "predq", "defatt", "defeph", "defq"],
    tint: list,
    mms_id: Union[str, int],
    data_path: Optional[str] = "",
):
    r"""Download files from MMS SDC.

    Download ancillary files containing field `var_str` over the time interval `tint`
    for the spacecraft `mms_id` to `data_path`.

    Parameters
    ----------
    product : {"predatt", "predeph", "predq", "defatt", "defeph", "defq"}
        Ancillary type.
    tint : list
        Time interval
    mms_id : str or int
        Spacecraft index (not used for "defq" and "predq", which cover the
        four spacecraft: ancillary/mms/defq).
    data_path : str, Optional
        Path of MMS data. Default uses the local path of the MMS configuration
        (`pyrfu.mms.MMS_CFG_PATH`, set with `mms.db_init`).

    """
    # List files in MMS SDC that match the request
    files_in_interval = list_files_ancillary_sdc(tint, mms_id, product)

    # Start session on MMS SDC ("public" or "sitl")
    sdc_session, headers, _ = _login_lasp()

    for file in files_in_interval:
        # Create local path following tree structure for the CDF files
        out_file = _make_path_local(file, product, mms_id, data_path)

        logger.info(
            "Downloading %s from %s...", os.path.basename(out_file), file["url"]
        )

        _download_file(
            sdc_session, file["url"], headers, out_file, file.get("file_size")
        )
