#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import logging
import re

# 3rd party imports
import numpy as np
import xarray as xr
from pycdfpp import load

# Local imports
from ..pyrf.ts_append import ts_append
from .db_get_ts import _resolve_source
from .get_data import _check_times, _get_file_content_sources, _list_files_sources
from .get_ts import get_ts

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

logger = logging.getLogger(__name__)


def _eis_names(cdf, pref, specie, data_unit, file_name):
    r"""Names of the variables of `specie` in a file (the P# of the look-up
    table changes with the file version), and its energy table."""
    names = [name for name, _ in cdf.items()]

    # Files not reprocessed (v3.0.0, 2015-2016) name the variables without the
    # data rate and level (e.g., mms1_epd_eis_extof_proton_P2_flux_t0)
    if not any(name.startswith(f"{pref}_") for name in names):
        raise ValueError(
            f"{file_name} uses the old EIS variable names: use the reprocessed "
            "(v*.100) files"
        )

    pattern = re.compile(rf"^{pref}_{specie}_P(\d+)_{data_unit}_t0$")
    p_nums = [match.group(1) for match in map(pattern.match, names) if match]

    if not p_nums:
        species = {
            match.group(1)
            for match in map(re.compile(rf"^{pref}_(\w+)_P\d+_\w+_t0$").match, names)
            if match
        }
        raise ValueError(
            f"No {specie} {data_unit} in {file_name}: species in the file are "
            f"{', '.join(sorted(species - {'dump'}))}"
        )

    cdf_names = {
        "spin": f"{pref}_spin",
        "sector": f"{pref}_sector",
        **{
            f"t{i:d}": f"{pref}_{specie}_P{p_nums[0]}_{data_unit}_t{i:d}"
            for i in range(6)
        },
        **{f"look_t{i:d}": f"{pref}_look_t{i:d}" for i in range(6)},
    }

    # Non-record-varying energy table of the telescope 0, one value per channel
    energy = {
        k: np.ravel(cdf[f"{pref}_{specie}_t0_energy{k}"].values)
        for k in ["", "_dminus", "_dplus"]
    }

    return cdf_names, energy


def get_eis_allt(
    tar_var,
    tint,
    mms_id,
    verbose: bool = True,
    data_path: str = "",
    source: str = "default",
):
    r"""Read energy spectrum of the selected specie in the selected energy
    range for all telescopes.

    Parameters
    ----------
    tar_var : str
        Key of the target variable like
        {data_unit}_{dtype}_{specie}_{data_rate}_{data_lvl}.
    tint : list of str
        Time interval.
    mms_id : int or float or str
        Index of the spacecraft.
    verbose : bool, Optional
        Set to True to follow the loading. Default is True.
    data_path : str, Optional
        Path of MMS data.
    source : {"default", "local", "sdc", "aws"}, Optional
        Resource to fetch the data from. Default uses default in
        `pyrfu/mms/config.json`. Each file is read (downloaded) once for all the
        telescopes.

    Returns
    -------
    out : xarray.Dataset
        Dataset containing the energy spectrum of the 6 telescopes of the
        Energy Ion Spectrometer.

    Raises
    ------
    ValueError
        If the data unit is not one of flux, counts and cps, if the files don't
        have the specie (e.g., helium in PHxTOF), if they are not reprocessed
        (v3.0.0, 2015-2016) or if the energy table changes in the time
        interval (split the time interval).

    Notes
    -----
    The variable names are read from each file: the P# of the look-up table
    is not always the major version of the file (e.g., P3 in v2.1.100). The
    specie "alpha" is read as "helium".

    Examples
    --------
    >>> from pyrfu import mms

    Define time interval

    >>> tint_brst = ["2017-07-23T16:54:24.000", "2017-07-23T17:00:00.000"]

    Read proton energy spectrum for all EIS telescopes

    >>> eis_allt = mms.get_eis_allt("Flux_extof_proton_srvy_l2", tint_brst, 2)

    """

    # Convert mms_id to integer
    mms_id = int(mms_id)

    data_unit, data_type, specie, data_rate, data_lvl = tar_var.split("_")
    data_unit = data_unit.lower()

    if data_unit not in ["flux", "counts", "cps"]:
        raise ValueError("Invalid data unit")

    if specie == "alpha":
        specie = "helium"

    var = {"inst": "epd-eis", "tmmode": data_rate, "lev": data_lvl, "dtype": data_type}
    pref = f"mms{mms_id:d}_epd_eis_{data_rate}_{data_lvl}_{data_type}"

    resource = _resolve_source(source)
    file_names, sdc_session, headers = _list_files_sources(
        resource, tint, str(mms_id), var, data_path
    )

    if not file_names:
        raise FileNotFoundError("no files for these inputs!!")

    outdict, energy = {}, None

    for file_name in file_names:
        file_content = _get_file_content_sources(
            resource, file_name, sdc_session, headers
        )
        cdf_names, file_energy = _eis_names(
            load(file_content), pref, specie, data_unit, file_name
        )

        if energy is None:
            energy = file_energy
        elif not np.array_equal(energy[""], file_energy[""]):
            raise ValueError(
                f"The energy table changes in {file_name}: split the time interval"
            )

        for key, cdf_name in cdf_names.items():
            if verbose:
                logger.info("Loading %s...", cdf_name)

            ts = get_ts(file_content, cdf_name, tint)
            outdict[key] = ts_append(outdict[key], ts) if key in outdict else ts

    outdict = {key: _check_times(value) for key, value in outdict.items()}

    for i in range(6):
        outdict[f"t{i:d}"] = outdict[f"t{i:d}"].rename({"Energy": "energy"})

    glob_attrs = {
        "delta_energy_plus": energy["_dplus"],
        "delta_energy_minus": energy["_dminus"],
        "species": specie,
        **outdict["spin"].attrs["GLOBAL"],
    }

    # Build Dataset
    out = xr.Dataset(outdict, attrs=glob_attrs)

    return out
