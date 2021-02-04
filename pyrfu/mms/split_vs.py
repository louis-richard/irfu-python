#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

import warnings


def split_vs(var_str):
    """Parse the variable keys.

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

    all_params_scalars = ["ni", "nbgi", "pbgi", "partni", "ne", "pbge", "nbge", "partne", "tsi",
                          "tperpi",
                          "tparai", "parttperpi", "parttparai", "tse", "tperpe", "tparae",
                          "parttperpe", "parttparae",
                          "pde", "pdi", "pderre", "pderri", "v", "v6", "defi", "defbgi", "defe",
                          "defbge",
                          "energyi", "bnergye", "epar", "sdev12", "sdev34", "flux-amb-pm2",
                          "padlowene", "padmidene",
                          "padhighene", "bpsd", "epsd"]

    all_params_vectors = ["r", "sti", "vi", "errvi", "partvi", "ste", "ve", "errve", "partve", "qi",
                          "errqi", "qe",
                          "errqe", "b", "e", "e2d", "es12", "es34"]

    all_params_tensors = ["pi", "partpi", "pe", "partpe", "ti", "partti", "te", "partte"]

    hpca_params_scalars = ["nhplus", "nheplus", "nheplusplus", "noplus", "tshplus", "tsheplus",
                           "tsheplusplus", "tsoplus",
                           "phase", "adcoff", "fluxhplus", "fluxheplus", "fluxheplusplus",
                           "fluxoplus"]

    hpca_params_tensors = ["vhplus", "vheplus", "vheplusplus", "voplus", "phplus", "pheplus",
                           "pheplusplus", "poplus",
                           "thplus", "theplus", "theplusplus", "toplus"]

    instruments = ["mec", "fpi", "edp", "edi", "hpca", "fgm", "dfg", "afg", "scm", "fsm", "dsp"]

    coordinate_systems = ["gse", "gsm", "dsl", "dbcs", "dmpa", "ssc", "bcs", "par"]

    data_lvls = ["ql", "sitl", "l1b", "l2a", "l2pre", "l2", "l3"]

    param = splitted_key[0]

    if param.lower() in all_params_scalars or param.lower() in hpca_params_scalars:
        tensor_order = 0
    elif param.lower() in all_params_vectors or param.lower() in hpca_params_tensors:
        tensor_order = 1
    elif param.lower() in all_params_tensors:
        tensor_order = 2
    else:
        raise ValueError(f"invalid PARAM : {param}")

    coordinate_system = []
    idx = 0

    if tensor_order > 0:
        coordinate_system = splitted_key[idx + 1]
        assert coordinate_system in coordinate_systems, "invalid COORDINATE_SYS"
        idx += 1

    instrument = splitted_key[idx + 1]
    assert instrument in instruments, "invalid INSTRUMENT"

    idx += 1

    tmmode = splitted_key[idx + 1]
    idx += 1

    if tmmode not in ["brst", "fast", "slow", "srvy"]:
        tmmode = "fast"
        idx -= 1
        warnings.warn("assuming TM_MODE = FAST", UserWarning)

    if len(splitted_key) == idx + 1:
        data_lvl = "l2"  # default
    else:
        data_lvl = splitted_key[idx + 1]
        assert data_lvl in data_lvls, "invalid DATA_LEVEL level"

    res = {"param": param, "to": tensor_order, "cs": coordinate_system, "inst": instrument,
           "tmmode": tmmode, "lev": data_lvl}

    return res
