import math
import numpy as np
import xarray as xr
from astropy.time import Time

from .get_feeps_active_eyes import get_feeps_active_eyes


def feeps_pitch_angles(inp_dset=None, b_bcs=None):
	"""
	Computes the FEEPS pitch angles for each telescope from magnetic field data.
	"""
	var = inp_dset.attrs
	mms_id = var["mmsId"]
	times = inp_dset.time
	btimes = b_bcs.time

	trange = Time(np.hstack([times.data.min(), times.data.max()]), format="datetime64").isot

	eyes = get_feeps_active_eyes(var, trange, var["mmsId"])

	idx_maps = None
	nbins = 13  # number of pitch angle bins; 10 deg = 17 bins, 15 deg = 13 bins
	dpa = 180.0 / nbins  # delta-pitch angle for each bin

	# Rotation matrices for FEEPS coord system (FCS) into body coordinate system (BCS):
	t_top = [[1./np.sqrt(2.), -1./np.sqrt(2.), 0], [1./np.sqrt(2.), 1./np.sqrt(2.), 0], [0, 0, 1]]
	t_bot = [[-1./np.sqrt(2.), -1./np.sqrt(2.), 0], [-1./np.sqrt(2.), 1./np.sqrt(2.), 0], [0, 0, -1]]

	# Telescope vectors in FCS:
	v_fcs = {"1": [0.347, -0.837, 0.423], "2": [0.347, -0.837, -0.423], "3": [0.837, -0.347, 0.423],
			"4": [0.837, -0.347, -0.423], "5": [-0.087, 0.000, 0.996], "6": [0.104, 0.180, 0.978],
			"7": [0.654, -0.377, 0.656], "8": [0.654, -0.377, -0.656], "9": [0.837, 0.347, 0.423],
			"10": [0.837, 0.347, -0.423], "11": [0.347, 0.837, 0.423], "12": [0.347, 0.837, -0.423]}


	telescope_map = {}
	telescope_map["bottom-electron"], telescope_map["bottom-ion"] = [[1, 2, 3, 4, 5, 9, 10, 11, 12], [6, 7, 8]]

	telescope_map["top-electron"], telescope_map["top-ion"] = [[1, 2, 3, 4, 5, 9, 10, 11, 12], [6, 7, 8]]

	top_tele_idx_map, bot_tele_idx_map = [{}, {}]

if var["dtype"] == "electron":
	pas = np.empty([len(btimes), 18])  # pitch angles for each eye at each time

	# Telescope vectors in Body Coordinate System:
	#   Factors of -1 account for 180 deg shift between particle velocity and telescope normal direction:
	# Top:
	vt_bcs, vb_bcs = [{}, {}]

	for s in telescope_map["top-{}".format(var["dtype"])]:
		s = str(s)

		vt_bcs[s] = [-1. * (t_top[0][0] * v_fcs[s][0] + t_top[0][1] * v_fcs[s][1] + t_top[0][2] * v_fcs[s][2]),
					 -1. * (t_top[1][0] * v_fcs[s][0] + t_top[1][1] * v_fcs[s][1] + t_top[1][2] * v_fcs[s][2]),
					 -1. * (t_top[2][0] * v_fcs[s][0] + t_top[2][1] * v_fcs[s][1] + t_top[2][2] * v_fcs[s][2])]

	for s in telescope_map["bottom-{}".format(var["dtype"])]:
		s = str(s)

		vb_bcs[s] = [-1. * (t_bot[0][0] * v_fcs[s][0] + t_bot[0][1] * v_fcs[s][1] + t_bot[0][2] * v_fcs[s][2]),
					 -1. * (t_bot[1][0] * v_fcs[s][0] + t_bot[1][1] * v_fcs[s][1] + t_bot[1][2] * v_fcs[s][2]),
					 -1. * (t_bot[2][0] * v_fcs[s][0] + t_bot[2][1] * v_fcs[s][1] + t_bot[2][2] * v_fcs[s][2])]

	for i, k in zip(np.arange(18),
					np.hstack([telescope_map["bottom-{}".format(var["dtype"])], telescope_map["top-{}".format(var["dtype"])]])):
		if i < 8:
			v_bcs = vt_bcs[str(k)]
		else:
			v_bcs = vb_bcs[str(k)]
		"""
		pas[:, i] = 180. / math.pi * np.arccos(
			(v_bcs[0] * b_bcs[:, 0] + v_bcs[1] * b_bcs[:, 1] + v_bcs[2] * b_bcs[:, 2]) / (
						np.sqrt(v_bcs[0] ** 2 + v_bcs[1] ** 2 + v_bcs[2] ** 2) * np.sqrt(
					b_bcs[:, 0] ** 2 + b_bcs[:, 1] ** 2 + b_bcs[:, 2] ** 2)))
		"""

		if var["tmmode"] == "srvy":
			if i < 8:
				top_tele_idx_map[k] = i
			else:
				bot_tele_idx_map[k] = i

			top_idxs, bot_idxs = [[], []]

			# PAs for only active eyes
			new_pas = np.empty(
				[len(btimes), len(eyes["top"]) + len(eyes["bottom"])])  # pitch angles for each eye at eaceh time

			for top_idx, top_eye in enumerate(eyes["top"]):
				new_pas[:, top_idx] = pas[:, top_tele_idx_map[top_eye]]
				top_idxs.append(top_idx)

			for bot_idx, bot_eye in enumerate(eyes["bottom"]):
				new_pas[:, bot_idx + len(eyes["top"])] = pas[:, bot_tele_idx_map[bot_eye]]
				bot_idxs.append(bot_idx + len(eyes["top"]))

			idx_maps = {"electron-top": top_idxs, "electron-bottom": bot_idxs}

		else:
			new_pas = pas




	elif Var["dtype"] == "ion":
		pas = np.empty([len(btimes), 6]) # pitch angles for each eye at each time

		# Telescope vectors in Body Coordinate System:
		#   Factors of -1 account for 180 deg shift between particle velocity and telescope normal direction:
		# Top:
		Vt6bcs = [-1.*(Ttop[0][0]*V6fcs[0] + Ttop[0][1]*V6fcs[1] + Ttop[0][2]*V6fcs[2]),\
					-1.*(Ttop[1][0]*V6fcs[0] + Ttop[1][1]*V6fcs[1] + Ttop[1][2]*V6fcs[2]),\
					-1.*(Ttop[2][0]*V6fcs[0] + Ttop[2][1]*V6fcs[1] + Ttop[2][2]*V6fcs[2])]

		Vt7bcs = [-1.*(Ttop[0][0]*V7fcs[0] + Ttop[0][1]*V7fcs[1] + Ttop[0][2]*V7fcs[2]),\
					-1.*(Ttop[1][0]*V7fcs[0] + Ttop[1][1]*V7fcs[1] + Ttop[1][2]*V7fcs[2]),\
					-1.*(Ttop[2][0]*V7fcs[0] + Ttop[2][1]*V7fcs[1] + Ttop[2][2]*V7fcs[2])]

		Vt8bcs = [-1.*(Ttop[0][0]*V8fcs[0] + Ttop[0][1]*V8fcs[1] + Ttop[0][2]*V8fcs[2]),\
					-1.*( Ttop[1][0]*V8fcs[0] + Ttop[1][1]*V8fcs[1] + Ttop[1][2]*V8fcs[2]),\
					-1.*(Ttop[2][0]*V8fcs[0] + Ttop[2][1]*V8fcs[1] + Ttop[2][2]*V8fcs[2])]

		# Bottom:
		Vb6bcs = [-1.*(Tbot[0][0]*V6fcs[0] + Tbot[0][1]*V6fcs[1] + Tbot[0][2]*V6fcs[2]),\
					-1.*(Tbot[1][0]*V6fcs[0] + Tbot[1][1]*V6fcs[1] + Tbot[1][2]*V6fcs[2]),\
					-1.*( Tbot[2][0]*V6fcs[0] + Tbot[2][1]*V6fcs[1] + Tbot[2][2]*V6fcs[2])]

		Vb7bcs = [-1.*(Tbot[0][0]*V7fcs[0] + Tbot[0][1]*V7fcs[1] + Tbot[0][2]*V7fcs[2]),\
					-1.*(Tbot[1][0]*V7fcs[0] + Tbot[1][1]*V7fcs[1] + Tbot[1][2]*V7fcs[2]),\
					-1.*(Tbot[2][0]*V7fcs[0] + Tbot[2][1]*V7fcs[1] + Tbot[2][2]*V7fcs[2])]

		Vb8bcs = [-1.*(Tbot[0][0]*V8fcs[0] + Tbot[0][1]*V8fcs[1] + Tbot[0][2]*V8fcs[2]),\
					-1.*(Tbot[1][0]*V8fcs[0] + Tbot[1][1]*V8fcs[1] + Tbot[1][2]*V8fcs[2]),\
					-1.*(Tbot[2][0]*V8fcs[0] + Tbot[2][1]*V8fcs[1] + Tbot[2][2]*V8fcs[2])]

		for i in range(0, 6):
			if i == 0:	Vbcs = Vt6bcs
			if i == 1:	Vbcs = Vt7bcs
			if i == 2:	Vbcs = Vt8bcs
			if i == 3:	Vbcs = Vb6bcs
			if i == 4:	Vbcs = Vb7bcs
			if i == 5:	Vbcs = Vb8bcs
			pas[:, i] = 180./math.pi*np.arccos((Vbcs[0]*Bbcs[:, 0] + Vbcs[1]*Bbcs[:, 1] + Vbcs[2]*Bbcs[:, 2])/(np.sqrt(Vbcs[0]**2+Vbcs[1]**2+Vbcs[2]**2) * np.sqrt(Bbcs[:, 0]**2+Bbcs[:, 1]**2+Bbcs[:, 2]**2)))

			# the following 2 hash tables map TOP/BOTTOM telescope #s to index of the PA array created above
			top_tele_idx_map = {}
			bot_tele_idx_map = {}
			top_tele_idx_map[6] = 0
			top_tele_idx_map[7] = 1
			top_tele_idx_map[8] = 2
			bot_tele_idx_map[6] = 3
			bot_tele_idx_map[7] = 4
			bot_tele_idx_map[8] = 5

			top_idxs = []
			bot_idxs = []

			# PAs for only active eyes
			new_pas = np.empty([len(btimes), len(eyes["top"])+len(eyes["bottom"])]) # pitch angles for each eye at eaceh time

			for top_idx, top_eye in enumerate(eyes["top"]):
				new_pas[:, top_idx] = pas[:, top_tele_idx_map[top_eye]]
				top_idxs.append(top_idx)

			for bot_idx, bot_eye in enumerate(eyes["bottom"]):
				new_pas[:, bot_idx+len(eyes["top"])] = pas[:, bot_tele_idx_map[bot_eye]]
				bot_idxs.append(bot_idx+len(eyes["top"]))


	outdata = xr.DataArray(new_pas,coords=[btimes,np.arange(18)],dims=["time","idx"])

	# interpolate to the PA time stamps
	out = outdata.interp({'time': times})

	return out