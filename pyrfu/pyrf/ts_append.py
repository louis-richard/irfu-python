import numpy as np
import xarray as xr


def ts_append(inp1=None, inp2=None):
	"""
	Concatenate two time series along the time axis

	Note : the time series have to be in the correct time order

	Parameters :
		inp1 : DataArray
			Time series of the first input (early times)

		inp2 DataArray
			Time series of the second input (late times)

	Returns :
		out : DataArray
			Concatenated time series

	TODO : replace "name" + i (ugly) with list of dims and coords
	"""

	if inp1 is None:
		raise ValueError("ts_append requires at least two arguments")
	
	if inp2 is None:
		raise ValueError("ts_append requires at least two arguments")
		
	if not isinstance(inp1, xr.DataArray):
		raise TypeError("inp1 must be a DataArray")
		
	if not isinstance(inp2, xr.DataArray):
		raise TypeError("inp1 must be a DataArray")
		
	outdata = {}

	if inp1.data.ndim != 1:
		outdata["data"] = np.vstack([inp1, inp2])

	else:
		outdata["data"] = np.hstack([inp1, inp2])

	outdata["attrs"] = {}

	for k in inp1.attrs:
		if isinstance(inp1.attrs[k], np.ndarray):
			outdata["attrs"][k] = np.hstack([inp1.attrs[k], inp2.attrs[k]])

		else:
			outdata["attrs"][k] = inp1.attrs[k]

	# get coordinates
	for i, dim in enumerate(inp1.dims):
		exec("dim" + str(i) + " = {}")

		if i == 0:
			# append time and time errors
			exec("dim" + str(i) + "['data'] = np.hstack([inp1." + str(dim) + ".data, inp2." + str(dim) + ".data])")

			# add attributes
			exec("dim"+str(i)+"['attrs'] = {}")

			for k in eval("inp1." + dim + ".attrs"):

				# if attrs is array time append
				if isinstance(eval("inp1." + dim + ".attrs[k]"), np.ndarray):
					exec("dim" + str(i) + "['attrs'][k] = np.hstack([inp1." + dim + ".attrs[k], inp2." + dim \
						 + ".attrs[k]])")

				else:
					exec("dim" + str(i) + "['attrs'][k] = inp1." + dim + ".attrs[k]")

		else:
			# Use values of other coordinates of inp1 assuming equal to inp2
			exec("dim" + str(i) + "['data'] = inp1." + str(dim) + ".data")

			# add attributes
			exec("dim" + str(i) + "['attrs'] = {}")

			for k in eval("inp1." + dim + ".attrs"):
				exec("dim" + str(i) + "['attrs'][k] = inp1." + dim + ".attrs[k]")

	# Prepare coords and dims to build DataArray
	dims, coords = [inp1.dims, [None] * len(inp1.dims)]

	for i in range(len(dims)):
		coords[i] = eval("dim" + str(i) + "['data']")

	# Create DataArray
	out = xr.DataArray(outdata["data"], coords=coords, dims=dims, attrs=outdata["attrs"])
	
	for i, dim in enumerate(dims):
		exec("out." + dim + ".attrs = dim" + str(i) + "['attrs']")

	return out
