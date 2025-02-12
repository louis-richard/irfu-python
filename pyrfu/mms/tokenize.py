#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

# Built-in imports
import os
from typing import Mapping

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "2.4.13"
__status__ = "Prototype"


ALL_PARAMS_SCALARS = [
    "ni",
    "nbgi",
    "pbgi",
    "partni",
    "ne",
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
    "bnergye",
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
    "pe",
    "partpe",
    "ti",
    "partti",
    "te",
    "partte",
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
    assert param in ALL_PARAMETERS, f"invalid PARAM : {param}"

    if param in PARAMS_SCALARS:
        tensor_order = 0
    elif param in ALL_PARAMS_VECTORS:
        tensor_order = 1
    else:
        tensor_order = 2

    return tensor_order


def tokenize(var_str: str) -> Mapping[str, str]:
    r"""Parse the variable keys.

    Parameters
    ----------
    var_str : str
        Input key of variable

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

    """
    splitted_key = var_str.split("_")
    assert len(splitted_key) == 4 or len(splitted_key) == 5

    tensor_order = _tensor_order(splitted_key)

    if len(splitted_key) == 5 and tensor_order > 0:
        parameter, coordinates_system, instrument, data_rate, data_level = splitted_key
    else:
        parameter, instrument, data_rate, data_level = splitted_key
        coordinates_system = ""

    # Parameter
    if parameter not in ALL_PARAMETERS:
        raise ValueError("invalid parameter")

    # Coordinate system
    if coordinates_system not in COORDINATE_SYSTEMS:
        raise ValueError("invalid coord. sys.")

    # Instrument
    if instrument not in INSTRUMENTS:
        raise ValueError("invalid instrument")

    # Sampling rate
    if data_rate not in SAMPLING_RATES:
        raise ValueError("invalid sampling mode")

    # Data level
    if data_level not in DATA_LVLS:
        raise ValueError("invalid data level")

    root_path = os.path.dirname(os.path.abspath(__file__))

    with open(
        os.sep.join([root_path, "mms_keys.json"]), "r", encoding="utf-8"
    ) as json_file:
        keys_ = json.load(json_file)

    res = {
        "param": splitted_key[0],
        "cs": coordinates_system,
        "inst": instrument,
        "tmmode": data_rate,
        "lev": data_level,
        "dtype": keys_[instrument][var_str.lower()]["dtype"],
        "cdf_name": keys_[instrument][var_str.lower()]["cdf_name"],
    }

    return res
