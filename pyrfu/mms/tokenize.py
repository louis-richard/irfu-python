#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2023"
__license__ = "MIT"
__version__ = "2.4.2"
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
]

ALL_PARAMS_VECTORS = [
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
]


SAMPLING_RATES = ["brst", "fast", "slow", "srvy"]

DATA_LVLS = ["ql", "sitl", "l1b", "l2a", "l2pre", "l2", "l3"]


def _tensor_order(splitted_key):
    param = splitted_key[0].lower()
    assert param in ALL_PARAMETERS, f"invalid PARAM : {param}"

    if param in PARAMS_SCALARS:
        tensor_order = 0
    elif param in ALL_PARAMS_VECTORS:
        tensor_order = 1
    else:
        tensor_order = 2

    return tensor_order


def tokenize(var_str):
    r"""Parses the variable keys.

    Parameters
    ----------
    var_str : str
        Input key of variable

    Returns
    -------
    out : dict
        Hash table containing :
            * param : Variable key.
            * to : Tensor order.
            * cs : Coordinate system.
            * inst : Instrument.
            * tmmode : Time mode.
            * lev" : Level of data.

    """

    splitted_key = var_str.split("_")
    assert len(splitted_key) == 4 or len(splitted_key) == 5

    tensor_order = _tensor_order(splitted_key)

    if len(splitted_key) == 5 and tensor_order > 0:
        parameter, coordinates_system, instrument, data_rate, data_level = splitted_key
        assert coordinates_system in COORDINATE_SYSTEMS, "invalid coord. sys."
    else:
        parameter, instrument, data_rate, data_level = splitted_key
        coordinates_system = []

    assert parameter in ALL_PARAMETERS, "invalid parameter"

    # Instrument
    assert instrument in INSTRUMENTS, "invalid INSTRUMENT"

    # Sampling rate
    assert data_rate in SAMPLING_RATES, "invalid sampling mode"

    # Data level
    assert data_level in DATA_LVLS, "invalid DATA_LEVEL level"

    res = {
        "param": splitted_key[0],
        "to": tensor_order,
        "cs": coordinates_system,
        "inst": instrument,
        "tmmode": data_rate,
        "lev": data_level,
    }

    return res
