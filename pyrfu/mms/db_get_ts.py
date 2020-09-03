# -*- coding: utf-8 -*-
"""
db_get_ts.py

@author : Louis RICHARD
"""

from .list_files import list_files
from .get_ts import get_ts
from ..pyrf import ts_append


# noinspection PyUnboundLocalVariable
def db_get_ts(dset_name="", cdf_name="", trange=None):
	"""
	Get variable time series in the cdf file

	Parameters :
		dsetName : str
			Name of the dataset

		cdfName : str
			Name of the target field in cdf file

		trange : list of str
			Time interval

	Returns : 
		out : DataArray
			Time series of the target variable

	"""

	dset = dset_name.split("_")

	# Index of the MMS spacecraft
	probe = dset[0][-1]

	var = {"inst": dset[1], "tmmode": dset[2], "lev": dset[3]}

	try:
		var["dtype"] = dset[4]
	except IndexError:
		pass	

	files = list_files(trange, probe, var)

	for i, file in enumerate(files):
		temp = get_ts(file, cdf_name, trange)
		if i == 0:
			out = temp
		else:
			out = ts_append(out, temp)
			
	return out
