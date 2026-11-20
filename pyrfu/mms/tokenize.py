#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import functools
import json
import os
from collections.abc import Mapping

# Local imports
from pyrfu._core.deprecation import warn_deprecated

# Keys deprecated since 2.6.0: the HPCA velocities in GSM of the files are the
# DBCS velocities rotated from GSE to GSM as if DBCS were GSE (the rotation by
# the tilt of the spin axis, 2 to 4 degrees, is missing)
_DEPRECATED_KEYS = {
    f"v{species}_gsm_hpca_{data_rate}_l2": f"v{species}_dbcs_hpca_{data_rate}_l2"
    for species in ["hplus", "heplus", "heplusplus", "oplus"]
    for data_rate in ["srvy", "brst"]
}

# Parameters with keys in mms_keys.json. Left out: nbge, pbge, defbge (the
# FPI background is DIS only), tsi, tse (computed in irfu-matlab: the trace of
# ti_dbcs or te_dbcs / 3), epar (the e_par keys), and the irfu-matlab names
# of files of the MMS team, which could not be checked: energyi/e (FPI ql),
# es12/34 and adcoff (EDP l2a)
ALL_PARAMS_SCALARS = [
    "ni",
    "errni",
    "nbgi",
    "pbgi",
    "partni",
    "partidxi",
    "ne",
    "errne",
    "partne",
    "partidxe",
    "scpmeani",
    "scpmeane",
    "tperpi",
    "tparai",
    "parttperpi",
    "parttparai",
    "tperpe",
    "tparae",
    "parttperpe",
    "parttparae",
    "pde",
    "pdi",
    "pderre",
    "pderri",
    "pdphotoe",
    "pdtoffi",
    "pdtoffe",
    "avgf1countsi",
    "avgf1countse",
    "v",
    "v6",
    "vpsp",
    "defi",
    "defbgi",
    "defe",
    "defpare",
    "defantie",
    "defperpe",
    "defpxe",
    "defmxe",
    "defpye",
    "defmye",
    "defpze",
    "defmze",
    "defpxi",
    "defmxi",
    "defpyi",
    "defmyi",
    "defpzi",
    "defmzi",
    "sdev12",
    "sdev34",
    "flux-amb-pm2",
    "flux-amb",
    "flux-amb-alt-cc",
    "flux-amb-alt-oc",
    "flux-amb-alt-oob",
    "flux-amb-perp-ob",
    "counts-q0",
    "padlowene",
    "padmidene",
    "padhighene",
    "bpsd",
    "epsd",
    "hfesp",
    "ionc",
    "ionc1",
    "ionc2",
    "ebeam1",
    "ebeam2",
    "status",
    "triweight",
    "trirchisq",
    "flagi",
    "flage",
    "nextlowi",
    "nexthighi",
    "nextlowe",
    "nexthighe",
    "qualitye",
    "bitmaske",
    "qualityv",
    "bitmaskv",
    "flag",
    "mlt",
    "mlat",
    "ldip",
    "kp",
    "dst",
    "eclipseearth",
    "eclipsemoon",
    "tilt",
    "gmst",
    "geodlat",
    "geodlon",
    "geodh",
    "lvec",
    "zvec",
    "pvec",
    "lphase",
    "zphase",
    "pphase",
    "latlonfnt89d",
    "latlonfst89d",
    "losscnt89d",
    "losscst89d",
    "fltypet89d",
    "latlonfnt89q",
    "latlonfst89q",
    "losscnt89q",
    "losscst89q",
    "fltypet89q",
    "latlonfnts04d",
    "latlonfsts04d",
    "losscnts04d",
    "losscsts04d",
    "fltypets04d",
    "epsdx",
    "epsdy",
    "epsdz",
    "bpsdscm1",
    "bpsdscm2",
    "bpsdscm3",
    "hirange",
    "mode",
    "stemp",
    "etemp",
    "swde12",
    "swde34",
]

ALL_PARAMS_VECTORS = [
    "vel",
    "r",
    "sti",
    "vi",
    "errvi",
    "erre",
    "eres",
    "partvi",
    "ste",
    "ve",
    "errve",
    "partve",
    "qi",
    "errqi",
    "qe",
    "errqe",
    "bhati",
    "bhate",
    "b",
    "e",
    "hmfe",
    "e2d",
    "quat",
    "vdrift",
    "bschb",
    "bsweep",
    "errem",
    "errep",
    "errvdriftm",
    "errvdriftp",
    "rsun",
    "rmoon",
    "bmodt89d",
    "bfnt89d",
    "bfst89d",
    "bmint89d",
    "rfnt89d",
    "rfst89d",
    "rmint89d",
    "bmodt89q",
    "bfnt89q",
    "bfst89q",
    "bmint89q",
    "rfnt89q",
    "rfst89q",
    "rmint89q",
    "bmodts04d",
    "bfnts04d",
    "bfsts04d",
    "bmints04d",
    "rfnts04d",
    "rfsts04d",
    "rmints04d",
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
    "xformi",
    "xforme",
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
    "saz",
    "azimuth",
    "tparahplus",
    "tparaheplus",
    "tparaheplusplus",
    "tparaoplus",
    "tperphplus",
    "tperpheplus",
    "tperpheplusplus",
    "tperpoplus",
    "flaghplus",
    "flagheplus",
    "flagheplusplus",
    "flagoplus",
    "spinnum",
    "mcpvmon",
    "tofvmon",
    "tofcounts",
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
    "vparahplus",
    "vparaheplus",
    "vparaheplusplus",
    "vparaoplus",
    "vperphplus",
    "vperpheplus",
    "vperpheplusplus",
    "vperpoplus",
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
    "geo",
    "sm",
    "gse2000",
    "smpa",
    "ssl",
    "eci",
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


def _warn_deprecated_key(var_str: str, stacklevel: int = 2) -> None:
    r"""Warn (FutureWarning) if `var_str` is a deprecated key, pointing at the
    caller of the function that calls this one (`stacklevel` 2)."""

    var_key = var_str.lower()

    if var_key in _DEPRECATED_KEYS:
        warn_deprecated(
            f"key {var_str!r}",
            "2.6.0",
            "3.0.0",
            f"the DBCS velocity {_DEPRECATED_KEYS[var_key]!r} rotated with "
            "pyrfu.mms.dsl2gsm and the definitive attitude (defatt, "
            "pyrfu.mms.load_ancillary)",
            "The HPCA velocities in GSM of the files lack the rotation by the "
            "tilt of the spin axis (2 to 4 degrees).",
            stacklevel=stacklevel + 1,
        )


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

    Notes
    -----
    The keys of the HPCA velocities in GSM (``v{s}_gsm_hpca_{srvy,brst}_l2``)
    are deprecated, with a FutureWarning: those of the files lack the rotation
    by the tilt of the spin axis (see `pyrfu.mms.get_data`).

    .. deprecated:: 2.6.0
        The ``v{s}_gsm_hpca_*`` keys will be removed in pyrfu 3.0.0; rotate the
        DBCS velocities (``v{s}_dbcs_hpca_*``) with `pyrfu.mms.dsl2gsm`.

    """

    _warn_deprecated_key(var_str)

    out = _tokenize(var_str)

    return out


def _tokenize(var_str: str) -> Mapping[str, str]:
    r"""Parse the variable key `var_str`, without the deprecation warnings
    (`tokenize`)."""

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

    # A scalar has no coordinate system
    if len(splitted_key) == 5 and tensor_order == 0:
        raise ValueError(
            f"{splitted_key[0]!r} is a scalar: no coordinate system in {var_str!r}"
        )

    if len(splitted_key) == 5:
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
