#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
split_vs.py

@author : Louis RICHARD
"""

import warnings


def split_vs(var_str=""):
	"""
	Parse the variable keys

	Paramters :
		- var_str            [str]                   Input key of variable

	Returns :
		- out               [dict]                  Dictionary containing : 
														- ["param"]     Variable key
														- ["to"]        Tensor order
														- ["cs"]        Coordinate system
														- ["inst"]      Instrument
														- ["tmmode"]    Time mode
														- ["lev"]       Level of data
	"""

	if not var_str:
		raise ValueError("splitVs requires at least one argument")

	if not isinstance(var_str, str):
		raise TypeError("var_str must be a string")

	tk = var_str.split("_")
	n_tk = len(tk)

	if n_tk < 3 or n_tk > 5:
		raise ValueError("invalid STRING format")

	all_params_scal = ["ni", "nbgi", "pbgi", "partni", "ne", "pbge", "nbge", "partne", "tsi", "tperpi",
					   "tparai", "parttperpi", "parttparai", "tse", "tperpe", "tparae", "parttperpe", "parttparae",
					   "pde", "pdi", "pderre", "pderri", "v", "v6", "enfluxi", "enfluxbgi", "enfluxe", "enfluxbge",
					   "energyi", "bnergye", "epar", "sdev12", "sdev34", "flux-amb-pm2", "padlowene", "padmidene",
					   "padhighene", "bpsd", "epsd"]

	all_params_vect = ["r", "sti", "vi", "errvi", "partvi", "ste", "ve", "errve", "partve", "b", "e", "e2d", "es12",
					   "es34"]

	all_params_tens = ["pi", "partpi", "pe", "partpe", "ti", "partti", "te", "partte"]

	hpca_params_scal = ["nhplus", "nheplus", "nheplusplus", "noplus", "tshplus", "tsheplus", "tsheplusplus", "tsoplus",
						"phase", "adcoff"]

	hpca_params_tens = ["vhplus", "vheplus", "vheplusplus", "voplus", "phplus", "pheplus", "pheplusplus", "poplus",
						"thplus", "theplus", "theplusplus", "toplus"]

	param = tk[0]

	if param.lower() in all_params_scal:
		tensor_order = 0
	elif param.lower() in all_params_vect:
		tensor_order = 1
	elif param.lower() in all_params_tens:
		tensor_order = 2
	elif param.lower() in hpca_params_scal:
		tensor_order = 0
	elif param.lower() in hpca_params_tens:
		tensor_order = 1
	else:
		raise ValueError(f"invalid PARAM : {param}")

	coordinate_system = []
	idx = 0

	if tensor_order > 0:
		coordinate_system = tk[idx+1]

		idx += 1

		if coordinate_system not in ["gse", "gsm", "dsl", "dbcs", "dmpa", "ssc", "bcs", "par"]:
			raise ValueError("invalid COORDINATE_SYS")

	instrument = tk[idx+1]
	idx += 1

	if instrument not in ["mec", "fpi", "edp", "edi", "hpca", "fgm", "dfg", "afg", "scm", "fsm", "dsp"]:
		raise ValueError("invalid INSTRUMENT")

	tmmode = tk[idx+1]
	idx += 1

	if tmmode not in ["brst", "fast", "slow", "srvy"]:
		tmmode = "fast"
		idx -= 1
		warnings.warn("assuming TM_MODE = FAST", UserWarning)

	if len(tk) == idx + 1:
		data_lvl = "l2"  # default
	else:
		data_lvl = tk[idx+1]

		if data_lvl not in ["ql", "sitl", "l1b", "l2a", "l2pre", "l2", "l3"]:
			raise ValueError("invalid DATA_LEVEL level")

	res = {"param": param, "to": tensor_order, "cs": coordinate_system, "inst": instrument, "tmmode": tmmode,
		   "lev": data_lvl}

	return res
