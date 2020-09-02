import numpy as np
import xarray as xr

from .resample import resample
from .c_4_k import c_4_k
from .normalize import normalize
from .avg_4sc import avg_4sc
from .dot import dot
from .cross import cross


def c_4_grad(r_list=None, b_list=None, method="grad"):
	"""
	Calculate gradient of physical field using 4 spacecraft technique. 
	
	Parameters :
		r_list : list of DataArray
			Time series of the positions of the spacecraft

		b_list : list of DataArray
			Time series of the magnetic field at the corresponding positions

		method : str
			Method flag : 
				"grad" -> compute gradient (default)
				"div" -> compute divergence
				"curl" -> compute curl
				"bdivb" -> compute b.div(b)
				"curv" -> compute curvature

	Returns :
		- out : DataArray
			Time series of the derivative of the input field corresponding to the method

	Example :
		>>> from pyrfu import mms, pyrf
		>>> # Time interval
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft indices
		>>> ic = np.arange(1,5)
		>>> b = [mms.get_data("B_gse_fgm_srvy_l2",Tint,i) for i in ic]
		>>> r = [mms.get_data("R_gse",Tint,i) for i in ic]
		>>> grad_b = pyrf.c_4_grad(r,b,"grad")

	Reference : 
		ISSI book  Eq. 14.16, 14.17 p. 353

	See also : 
		c_4_k
	
	"""

	if (r_list is None) or (b_list is None):
		raise ValueError("c_4_grad requires at least 2 arguments")

	if not isinstance(r_list, list) or len(r_list) != 4:
		raise TypeError("R must be a list of 4SC position")

	if not isinstance(b_list, list) or len(b_list) != 4:
		raise TypeError("B must be a list of 4SC position")

	for i in range(4):
		if not isinstance(r_list[i], xr.DataArray):
			raise TypeError("Spacecraft position must be DataArray")

		if not isinstance(b_list[i], xr.DataArray):
			raise TypeError("Magnetic field must be DataArray")

	# Resample with respect to 1st spacecraft
	r_list = [resample(r, b_list[0]) for r in r_list]
	b_list = [resample(b, b_list[0]) for b in b_list]

	# Compute reciprocal vectors in barycentric coordinates (see c_4_k)
	k_list = c_4_k(r_list)

	# Magnetic field at the center of mass of the tetrahedron
	b_avg = avg_4sc(b_list)

	b_dict = {"1": b_list[0], "2": b_list[1], "3": b_list[2], "4": b_list[3]}
	k_dict = {"1": k_list[0], "2": k_list[1], "3": k_list[2], "4": k_list[3]}

	mms_list = b_dict.keys()

	# Gradient of scalar/vector
	if len(b_dict["1"].shape) == 1:
		grad_b = np.zeros((len(b_dict["1"]), 3))

		for mms_id in mms_list:
			grad_b += k_dict[mms_id].data * np.tile(b_dict[mms_id].data, (3, 1)).T

	else:
		grad_b = np.zeros((len(b_dict["1"]), 3, 3))

		for i in range(3):
			for j in range(3):
				for mms_id in mms_list:
					grad_b[:, j, i] += k_dict[mms_id][:, i].data * b_dict[mms_id][:, j].data

	# Gradient
	if method.lower() == "grad":
		outdata = grad_b

	# Divergence
	elif method.lower() == "div":
		div_b = np.zeros(len(b_dict["1"]))

		for mms_id in mms_list:
			div_b += dot(k_dict[mms_id], b_dict[mms_id]).data

		outdata = div_b
	
	# Curl
	elif method.lower() == "curl":
		curl_b = np.zeros((len(b_dict["1"]), 3))

		for mms_id in mms_list:
			curl_b += cross(k_dict[mms_id], b_dict[mms_id]).data

		outdata = curl_b
		
	# B.div(B)
	elif method.lower() == "bdivb":
		b_div_b = np.zeros(b_avg.shape)

		for i in range(3):
			b_div_b[:, i] = np.sum(b_avg.data * grad_b[:, i, :], axis=1)

		outdata = b_div_b
		
	# Curvature
	elif method.lower() == "curv":
		bhat_list = [normalize(b) for b in b_list]

		curv = c_4_grad(r_list, bhat_list, method="bdivb")

		outdata = curv.data

	else:
		raise ValueError("Invalid method")
		
	if len(outdata.shape) == 1:
		out = xr.DataArray(outdata, coords=[b_dict["1"].time], dims=["time"])

	elif len(outdata.shape) == 2:
		out = xr.DataArray(outdata, coords=[b_dict["1"].time, ["x", "y", "z"]], dims=["time", "comp"])

	elif len(outdata.shape) == 3:
		out = xr.DataArray(
			outdata, coords=[b_dict["1"].time, ["x", "y", "z"], ["x", "y", "z"]], dims=["time", "vcomp", "hcomp"])

	else:
		raise TypeError("Invalid type")

	return out
