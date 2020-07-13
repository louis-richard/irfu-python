from .list_files import list_files
from .get_ts import get_ts
from ..pyrf import ts_append


def db_get_ts(dsetName,cdfName,trange):
	Var = {}

	dset = dsetName.split("_")
	probe 			= dset[0][-1]
	Var["inst"] 	= dset[1]
	Var["tmmode"] 	= dset[2]
	Var["lev"] 		= dset[3]
	try:
		Var["dtype"] = dset[4]
	except IndexError:
		pass	

	files = list_files(trange,probe,Var)

	for i, file in enumerate(files):
		temp = get_ts(file,cdfName,trange)
		if i == 0:
			out = temp
		else :
			out = ts_append(out,temp)
	return out