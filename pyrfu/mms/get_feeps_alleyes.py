#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
get_feeps_alleyes.py

@author : Louis RICHARD
"""

import xarray as xr

from .get_feeps_oneeye import get_feeps_oneeye
from .get_feeps_active_eyes import get_feeps_active_eyes


def get_feeps_alleyes(tar_var="fluxe_brst_l2", tint=None, mms_id=1):

	specie = tar_var.split("_")[0][-1]

	var = {"tmmode": tar_var.split("_")[1], "lev": tar_var.split("_")[2], "mmsId": mms_id}

	if specie == "e":
		var["dtype"] = "electron"
	elif specie == "i":
		var["dtype"] = "ion"
	else:
		raise ValueError("Invalid specie")

	active_eyes = get_feeps_active_eyes(var, tint, mms_id)

	e_ids = [f"{k}-{s:d}" for k in active_eyes for s in active_eyes[k]]

	out_dict = {}

	for e_id in e_ids:
		out_dict[e_id] = get_feeps_oneeye(tar_var, e_id, tint, mms_id)

	out = xr.Dataset(out_dict)

	out.attrs = var

	out.attrs["species"] = "{}s".format(var["dtype"])

	return out
