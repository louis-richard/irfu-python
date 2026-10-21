#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Built-in imports
import functools
import json
import os
from typing import Mapping

__author__ = "Louis Richard"
__email__ = "louis.richard@physics.ox.ac.uk"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


ALL_PARAMS_SCALARS = [
    "ni",
    "errni",
    "nbgi",
    "pbgi",
    "partni",
    "ne",
    "errne",
    "pbge",
    "nbge",
    "partne",
    "tsi",
    "tperpi",
    "tparai",
    "parttperpi",
    "parttparai",
    "tse",
    "tperpe",
    "tparae",
    "parttperpe",
    "parttparae",
    "pde",
    "pdi",
    "pderre",
    "pderri",
    "v",
    "v6",
    "defi",
    "defbgi",
    "defe",
    "defbge",
    "energyi",
    "energye",
    "epar",
    "sdev12",
    "sdev34",
    "flux-amb-pm2",
    "padlowene",
    "padmidene",
    "padhighene",
    "bpsd",
    "epsd",
    "ionc",
]

ALL_PARAMS_VECTORS = [
    "vel",
    "r",
    "sti",
    "vi",
    "errvi",
    "erre",
    "partvi",
    "ste",
    "ve",
    "errve",
    "partve",
    "qi",
    "errqi",
    "qe",
    "errqe",
    "b",
    "e",
    "hmfe",
    "e2d",
    "es12",
    "es34",
]

ALL_PARAMS_TENSORS = [
    "pi",
    "partpi",
    "errpi",
    "pe",
    "partpe",
    "errpe",
    "ti",
    "partti",
    "errti",
    "te",
    "partte",
    "errte",
]

HPCA_PARAMS_SCALARS = [
    "nhplus",
    "nheplus",
    "nheplusplus",
    "noplus",
    "tshplus",
    "tsheplus",
    "tsheplusplus",
    "tsoplus",
    "dpfhplus",
    "dpfheplus",
    "dpfheplusplus",
    "dpfoplus",
    "phase",
    "adcoff",
    "saz",
    "azimuth",
]

HPCA_PARAMS_TENSORS = [
    "vhplus",
    "vheplus",
    "vheplusplus",
    "voplus",
    "phplus",
    "pheplus",
    "pheplusplus",
    "poplus",
    "thplus",
    "theplus",
    "theplusplus",
    "toplus",
]

PARAMS_SCALARS = [*ALL_PARAMS_SCALARS, *HPCA_PARAMS_SCALARS]
PARAMS_VECTORS = [*ALL_PARAMS_VECTORS]
PARAMS_TENSORS = [*ALL_PARAMS_TENSORS, *HPCA_PARAMS_TENSORS]
ALL_PARAMETERS = [*PARAMS_SCALARS, *PARAMS_VECTORS, *PARAMS_TENSORS]


COORDINATE_SYSTEMS = [
    "gse",
    "gsm",
    "dsl",
    "dbcs",
    "dmpa",
    "ssc",
    "bcs",
    "par",
    "",
]

INSTRUMENTS = [
    "mec",
    "fpi",
    "edp",
    "edi",
    "hpca",
    "fgm",
    "dfg",
    "afg",
    "scm",
    "fsm",
    "dsp",
    "aspoc",
]


SAMPLING_RATES = ["brst", "fast", "slow", "srvy"]

DATA_LVLS = ["ql", "sitl", "l1b", "l2a", "l2pre", "l2", "l3"]


def _tensor_order(splitted_key: list[str]) -> int:
    r"""Determine the tensor order of the variable.

    Parameters
    ----------
    splitted_key : list
        Variable key splitted by "_".

    Returns
    -------
    tensor_order : str
        Tensor order of the variable.

    """
    param = splitted_key[0].lower()

    if param in PARAMS_SCALARS:
        tensor_order = 0
    elif param in ALL_PARAMS_VECTORS:
        tensor_order = 1
    else:
        tensor_order = 2

    return tensor_order


@functools.lru_cache(maxsize=1)
def _mms_keys() -> dict:
    r"""The variable keys of mms_keys.json, read once."""
    root_path = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(root_path, "mms_keys.json"), "r", encoding="utf-8") as file:
        return json.load(file)


def tokenize(var_str: str) -> Mapping[str, str]:
    r"""Parse the variable keys.

    Parameters
    ----------
    var_str : str
        Input key of variable, param[_cs]_inst_rate_level, in any case (as
        the irfu-matlab names, e.g., "B_gse_fgm_srvy_l2").

    Returns
    -------
    out : dict
        Hash table containing :
            * param : Variable key.
            * cs : Coordinate system.
            * inst : Instrument.
            * tmmode : Time mode.
            * lev" : Level of data.
            * dtype: Data type.
            * cdf_name : variable name in the CDF file.

    Raises
    ------
    ValueError
        If the key does not have 4 or 5 parts, if a part is invalid, or if
        the variable is not supported (not listed in mms_keys.json).

    """
    # Case-insensitive, as the irfu-matlab names ("B_gse_fgm_srvy_l2")
    var_key = var_str.lower()
    splitted_key = var_key.split("_")

    if len(splitted_key) not in [4, 5]:
        raise ValueError(
            f"{var_str!r} is not a variable key: param[_cs]_inst_rate_level"
        )

    # Parameter (the first part in both forms)
    if splitted_key[0] not in ALL_PARAMETERS:
        raise ValueError(f"invalid parameter {splitted_key[0]!r} in {var_str!r}")

    tensor_order = _tensor_order(splitted_key)

    if len(splitted_key) == 5 and tensor_order > 0:
        parameter, coordinates_system, instrument, data_rate, data_level = splitted_key
    else:
        parameter, instrument, data_rate, data_level = splitted_key
        coordinates_system = ""

    # Coordinate system
    if coordinates_system not in COORDINATE_SYSTEMS:
        raise ValueError(f"invalid coord. sys. {coordinates_system!r} in {var_str!r}")

    # Instrument
    if instrument not in INSTRUMENTS:
        raise ValueError(f"invalid instrument {instrument!r} in {var_str!r}")

    # Sampling rate
    if data_rate not in SAMPLING_RATES:
        raise ValueError(f"invalid sampling mode {data_rate!r} in {var_str!r}")

    # Data level
    if data_level not in DATA_LVLS:
        raise ValueError(f"invalid data level {data_level!r} in {var_str!r}")

    keys_ = _mms_keys()

    # Valid parts do not make a supported variable: only the combinations
    # listed in mms_keys.json can be loaded
    if var_key not in keys_.get(instrument, {}):
        raise ValueError(
            f"{var_str} is not supported by pyrfu.mms.get_data (not in mms_keys.json)"
        )

    res = {
        "param": parameter,
        "cs": coordinates_system,
        "inst": instrument,
        "tmmode": data_rate,
        "lev": data_level,
        "dtype": keys_[instrument][var_key]["dtype"],
        "cdf_name": keys_[instrument][var_key]["cdf_name"],
    }

    return res
