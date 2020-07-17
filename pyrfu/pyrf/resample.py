import numpy as np
import warnings
import bisect
from scipy import interpolate
import xarray as xr




def resample(inp=None,ref=None,**kwargs):
	"""
	Resample inp to the time line of ref. If sampling of X is more than two times higher than Y, we average X, otherwise
	we interpolate X.
	
	Parameters :
		inp : DataArray
			Time series to resample

		ref : DataArray
			Reference time line

	Options :
		method : str
			Method of interpolation "spline", "linear" etc. (default "linear") if method is given then interpolate 
			independent of sampling.

		fs : float
			Sampling frequency of the Y signal, 1/window

		window : int/float/array
			Length of the averaging window, 1/fsample

		fs : str
			Sampling frequency of the Y signal, 1/window

		mean : bool
			Use mean when averaging

		median : bool
			Use median instead of mean when averaging

		max : bool
			Use max instead of mean when averaging

	Returns :
		out : DataArray
			Resampled input to the reference time line using the selected method
		
	"""

	haveoptions = False

	if not isinstance(inp,xr.DataArray):
		raise TypeError("Input must be a DataArray")

	if not isinstance(ref,xr.DataArray):
		raise TypeError("Reference must be a DataArray")

	if kwargs: have_options = True

	sfy         = []
	thresh      = 0
	method      = ""
	flag_do     = "check"
	median_flag = False
	mean_flag   = False
	max_flag    = False

	#pdb.set_trace()
	if "method" in kwargs:
		if isinstance(kwargs["method"],str):
			method = kwargs["method"]
		else :
			raise TypeError("METHOD must be string")

	if "fs" in kwargs:
		if sfy: raise ValueError("FSAMPLE/WINDOW already specified")

		if isinstance(kwargs["fs"],int) or isinstance(kwargs["fs"],float):
			sfy = kwargs["fs"]
		else :
			raise TypeError("FSAMPLE must be numeric")

	if "window" in kwargs:
		if sfy: raise ValueError("FSAMPLE/WINDOW already specified")

		if (isinstance(kwargs["window"],int)
				or isinstance(kwargs["window"],float)
				or isinstance(kwargs["window"],np.ndarray)):
			sfy = 1/kwargs["window"]
		else:
			raise TypeError("METHOD must be numeric")

	if "mean" in kwargs:
		if kwargs["mean"]:
			mean_flag   = True
			flag_do     = "average"

	if "median" in kwargs:
		if kwargs["median"]:
			median_flag = True
			flag_do     = "average"

	if "max" in kwargs:
		if kwargs["max"]:
			max_flag    = True
			flag_do     = "average"

	inp_time = inp.time.data.view("i8")*1e-9
	ref_time = ref.time.data.view("i8")*1e-9
	inp_data = inp.data
	
	if len(inp) == 1:
		if len(inp.shape) == 1:
			outdata = np.tile(inp_data,len(ref_time))
		else :
			outdata = np.tile(inp_data,(len(ref_time),1))

	ndata = len(ref_time)

	if flag_do == "check":
		if ndata > 1:
			if not sfy:
				sfy1 = 1/(ref_time[1]-ref_time[0])

				if ndata == 2:
					sfy = sfy1
				else:
					not_found = True

				cur     = 2
				MAXTRY  = 10

				while (not_found and cur <= ndata and cur-3 < MAXTRY):
					sfy = 1/(ref_time[cur]-ref_time[cur-1])

					if np.absolute(sfy-sfy1) < sfy*0.001:
						not_found   = 0
						sfy         = (sfy+sfy1)/2
						break

					sfy = sfy1
					cur += 1

				if not_found:
					sfy = sfy1
					raise RuntimeError("{Cannot guess sampling frequency. Tried {:d} times}".format(MAXTRY))
				
				del sfy1

			if len(inp_time) / (inp_time[-1] - inp_time[0]) > 2 * sfy:
				flag_do = "average"
				warnings.warn("Using averages in resamp",UserWarning)
			else:
				flag_do = "interpolation"
		else :
			flag_do = "interpolation"  # If one output time then do interpolation

	if flag_do == "average":
		if method:
			raise RunTimeError("cannot mix interpolation and averaging flags")

		if not sfy:
			sfy1 = 1/(ref_time[1]-ref_time[0])

			if ndata == 2:
				sfy = sfy1
			else:
				not_found = True
			
			cur = 2
			MAXTRY = 10
			
			while (not_found and cur <= ndata and cur-3 < MAXTRY):
				sfy = 1/(ref_time[cur]-ref_time[cur-1])
				
				if np.absolute(sfy-sfy1) < sfy*0.001:
					not_found   = 0
					sfy         = (sfy+sfy1)/2;
					break
				
				sfy = sfy1
				cur += 1

			if not_found:
				sfy = sfy1
				raise RuntimeError("{Cannot guess sampling frequency. Tried {:d} times}".format(MAXTRY))
			del sfy1

		dt2 = 0.5 / sfy  # Half interval

		#pdb.set_trace()
		inp_shape = list(inp_data.shape)
		inp_shape[0] = ndata

		inp_shape = tuple(inp_shape)

		outdata = np.zeros(inp_shape)
		"""
		try:
			outdata = np.zeros((ndata, inp_data.shape[1]))
		except IndexError:
			outdata = np.zeros(ndata)
		"""
		
		for i in range(ndata):
			idxl    = bisect.bisect_left(inp_time, ref_time[i] - dt2)
			idxr    = bisect.bisect_right(inp_time, ref_time[i] + dt2)
			ii      = np.arange(idxl, idxr)

			if ii.size == 0:
				outdata[i, ...] = np.nan
			else:
				if thresh:
					sdev    = np.std(inp_data[ii, ...])
					mm      = np.mean(inp_data[ii, ...])

					if any(not np.isnan(sdev)):
						for k in range(len(sdev)):
							if not np.isnan(sdev[k]):
								kk = bisect.bisect_right(inp_data[ii, k + 1] - mm[k], thresh * sdev[k])
								if kk:
									if median_flag:
										outdata[j, k + 1] = np.median(inp_data[ii[kk], k + 1],axis=0)
									elif max_flag:
										outdata[j, k + 1] = np.max(inp_data[ii[kk], k + 1],axis=0)
									else:
										outdata[j, k + 1] = np.mean(inp_data[ii[kk], k + 1],axis=0)
								else:
									outdata[i, k + 1] = np.nan
							else:
								outdata[i, ...] = np.nan
				else:
					if median_flag:
						outdata[i, ...] = np.median(inp_data[ii, ...],axis=0)
					elif max_flag:
						outdata[i, ...] = np.max(inp_data[ii, ...],axis=0)
					else:
						outdata[i, ...] = np.mean(inp_data[ii, ...],axis=0)

	elif flag_do == "interpolation":
		if any([mean_flag,median_flag,max_flag]):
			raise RunTimeError("cannot mix interpolation and averaging flags")

		if not method: method = "linear"

		# If time series agree, no interpolation is necessary.
		if len(inp_time) == len(ref_time) and all(inp_time == ref_time):
			outdata = inp_data
			coords  = [ref.coords["time"]]

			if len(inp.coords) > 1:
				for k in inp.dims[1:]:
					coords.append(inp.coords[k])

			out = xr.DataArray(outdata, coords=coords, dims=inp.dims,attrs=inp.attrs)
			
			return out

		tck     = interpolate.interp1d(inp_time, inp_data, kind=method, axis=0, fill_value="extrapolate")
		outdata = tck(ref_time)

	coords = [ref.coords["time"]]

	if len(inp.coords) > 1:
		for k in inp.dims[1:]:
			coords.append(inp.coords[k])

	out = xr.DataArray(outdata, coords=coords, dims=inp.dims,attrs=inp.attrs)
	
	return out