import os
import re
# debugger
import pdb
import bisect
# array modules
import numpy as np
import xarray as xr

# cdf module
from spacepy import pycdf
# physical constants
from astropy import constants
# signal porcessing, and optimizing
from scipy import interpolate, optimize, signal, fft

import warnings 

# Time modules
import datetime
from astropy.time import Time
from dateutil import parser


from .splitVs import splitVs
from .list_files import list_files
from .get_dist import get_dist
from .get_ts import get_ts
from ..pyrf import ts_append
from ..pyrf import dist_append


#-----------------------------------------------------------------------------------------------------------------------
def get_data(varStr="", tint=None, probe="1", silent=False):
	"""
	Load a variable. varStr must in var (see below)

	Parameters :
		- varStr            [xarray]                Key of the target variable
		- tint              [list]                  Time interval
		- probe             [str/int/float]         Index of the target probe
		- silent            [bool]                  Set to False (default) to follow the loading

	Returns :
		- out               [xarray]                Time serie of the target variable of measured by the target 
													spacecraft over the selected time interval
	
	Example :
		>>> # path of MMS data
		>>> data_path = "/Volumes/mms"
		>>> # Time interval
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # index of the spacecraft
		>>> ic = 1
		>>> Bxyz = pyrf.get_data("B_gse_fgm_brst_l2",Tint,ic)

	EPHEMERIS :
	"R_gse", "R_gsm"

	FGM : 
	"B_gsm_fgm_srvy_l2", "B_gsm_fgm_brst_l2", "B_gse_fgm_srvy_l2"
	"B_gse_fgm_brst_l2", "B_bcs_fgm_srvy_l2", "B_bcs_fgm_brst_l2"
	"B_dmpa_fgm_srvy_l2", "B_dmpa_fgm_brst_l2"

	DFG & AFG :
	"B_gsm_dfg_srvy_l2pre", "B_gse_dfg_srvy_l2pre", "B_dmpa_dfg_srvy_l2pre"
	"B_bcs_dfg_srvy_l2pre", "B_gsm_afg_srvy_l2pre", "B_gse_afg_srvy_l2pre"
	"B_dmpa_afg_srvy_l2pre", "B_bcs_afg_srvy_l2pre"

	SCM :
	"B_gse_scm_brst_l2"

	EDP :
	"Phase_edp_fast_l2a", "Phase_edp_slow_l2a", "Sdev12_edp_slow_l2a"
	"Sdev34_edp_slow_l2a", "Sdev12_edp_fast_l2a", "Sdev34_edp_fast_l2a"
	"E_dsl_edp_brst_l2"
	"E_dsl_edp_fast_l2", "E_dsl_edp_brst_ql", "E_dsl_edp_fast_ql"
	"E_dsl_edp_slow_l2", "E_gse_edp_brst_l2", "E_gse_edp_fast_l2"
	"E_gse_edp_slow_l2", "E2d_dsl_edp_brst_l2pre", "E2d_dsl_edp_fast_l2pre"
	"E2d_dsl_edp_brst_ql", "E2d_dsl_edp_fast_ql", "E2d_dsl_edp_l2pre"
	"E2d_dsl_edp_fast_l2pre", "E2d_dsl_edp_brst_l2pre", "E_dsl_edp_l2pre"
	"E_dsl_edp_fast_l2pre", "E_dsl_edp_brst_l2pre", "E_dsl_edp_slow_l2pre"
	"E_ssc_edp_brst_l2a", "E_ssc_edp_fast_l2a", "E_ssc_edp_slow_l2a"
	"V_edp_fast_sitl", "V_edp_slow_sitl", "V_edp_slow_l2"
	"V_edp_fast_l2", "V_edp_brst_l2"

	FPI Ions : 
	"Vi_dbcs_fpi_brst_l2", "Vi_dbcs_fpi_fast_l2", "Vi_dbcs_fpi_l2"
	"Vi_gse_fpi_ql", "Vi_gse_fpi_fast_ql", "Vi_dbcs_fpi_fast_ql"
	"Vi_gse_fpi_fast_l2", "Vi_gse_fpi_brst_l2", "partVi_gse_fpi_brst_l2"
	"Ni_fpi_brst_l2", "partNi_fpi_brst_l2", "Ni_fpi_brst"
	"Ni_fpi_fast_l2", "Ni_fpi_ql", "Enfluxi_fpi_fast_ql"
	"Enfluxi_fpi_fast_l2", "Tperpi_fpi_brst_l2", "Tparai_fpi_brst_l2"
	"partTperpi_fpi_brst_l2", "partTparai_fpi_brst_l2", "Ti_dbcs_fpi_brst_l2"
	"Ti_dbcs_fpi_brst", "Ti_dbcs_fpi_fast_l2", "Ti_gse_fpi_ql"
	"Ti_dbcs_fpi_ql", "Ti_gse_fpi_brst_l2", "Pi_dbcs_fpi_brst_l2"
	"Pi_dbcs_fpi_brst", "Pi_dbcs_fpi_fast_l2", "Pi_gse_fpi_ql"
	"Pi_gse_fpi_brst_l2"

	FPI Electrons :
	"Ve_dbcs_fpi_brst_l2", "Ve_dbcs_fpi_brst", "Ve_dbcs_fpi_ql"
	"Ve_dbcs_fpi_fast_l2", "Ve_gse_fpi_ql", "Ve_gse_fpi_fast_l2"
	"Ve_gse_fpi_brst_l2", "partVe_gse_fpi_brst_l2", "Enfluxe_fpi_fast_ql"
	"Enfluxe_fpi_fast_l2", "Ne_fpi_brst_l2", "partNe_fpi_brst_l2"
	"Ne_fpi_brst", "Ne_fpi_fast_l2", "Ne_fpi_ql"
	"Tperpe_fpi_brst_l2", "Tparae_fpi_brst_l2", "partTperpe_fpi_brst_l2"
	"partTparae_fpi_brst_l2", "Te_dbcs_fpi_brst_l2", "Te_dbcs_fpi_brst"
	"Te_dbcs_fpi_fast_l2", "Te_gse_fpi_ql", "Te_dbcs_fpi_ql"
	"Te_gse_fpi_brst_l2", "Pe_dbcs_fpi_brst_l2", "Pe_dbcs_fpi_brst"
	"Pe_dbcs_fpi_fast_l2", "Pe_gse_fpi_ql", "Pe_gse_fpi_brst_l2"

	HPCA : 
	"Nhplus_hpca_srvy_l2", "Nheplus_hpca_srvy_l2", "Nheplusplus_hpca_srvy_l2"
	"Noplus_hpca_srvy_l2", "Tshplus_hpca_srvy_l2", "Tsheplus_hpca_srvy_l2"
	"Tsheplusplus_hpca_srvy_l2", "Tsoplus_hpca_srvy_l2", "Vhplus_dbcs_hpca_srvy_l2"
	"Vheplus_dbcs_hpca_srvy_l2", "Vheplusplus_dbcs_hpca_srvy_l2", "Voplus_dbcs_hpca_srvy_l2"
	"Phplus_dbcs_hpca_srvy_l2", "Pheplus_dbcs_hpca_srvy_l2", "Pheplusplus_dbcs_hpca_srvy_l2"
	"Poplus_dbcs_hpca_srvy_l2", "Thplus_dbcs_hpca_srvy_l2", "Theplus_dbcs_hpca_srvy_l2"
	"Theplusplus_dbcs_hpca_srvy_l2", "Toplus_dbcs_hpca_srvy_l2", "Vhplus_gsm_hpca_srvy_l2"
	"Vheplus_gsm_hpca_srvy_l2", "Vheplusplus_gsm_hpca_srvy_l2", "Voplus_gsm_hpca_srvy_l2"
	"Nhplus_hpca_brst_l2", "Nheplus_hpca_brst_l2", "Nheplusplus_hpca_brst_l2"
	"Noplus_hpca_brst_l2", "Tshplus_hpca_brst_l2", "Tsheplus_hpca_brst_l2"
	"Tsheplusplus_hpca_brst_l2", "Tsoplus_hpca_brst_l2", "Vhplus_dbcs_hpca_brst_l2"
	"Vheplus_dbcs_hpca_brst_l2", "Vheplusplus_dbcs_hpca_brst_l2", "Voplus_dbcs_hpca_brst_l2"
	"Phplus_dbcs_hpca_brst_l2", "Pheplus_dbcs_hpca_brst_l2", "Pheplusplus_dbcs_hpca_brst_l2"
	"Poplus_dbcs_hpca_brst_l2", "Thplus_dbcs_hpca_brst_l2", "Theplus_dbcs_hpca_brst_l2"
	"Theplusplus_dbcs_hpca_brst_l2", "Toplus_dbcs_hpca_brst_l2", "Vhplus_gsm_hpca_brst_l2"
	"Vheplus_gsm_hpca_brst_l2", "Vheplusplus_gsm_hpca_brst_l2", "Voplus_gsm_hpca_brst_l2"
	"Phplus_gsm_hpca_brst_l2", "Pheplus_gsm_hpca_brst_l2", "Pheplusplus_gsm_hpca_brst_l2"
	"Poplus_gsm_hpca_brst_l2", "Thplus_gsm_hpca_brst_l2", "Theplus_gsm_hpca_brst_l2"
	"Theplusplus_gsm_hpca_brst_l2", "Toplus_gsm_hpca_brst_l2"
	"""

	




	if not varStr:
		raise ValueError("get_data requires at least 2 arguments")

	if tint is None:
		raise ValueError("get_data requires at least 2 arguments")


	if isinstance(probe,int) or isinstance(probe,float):
		probe = str(probe)

		
	# Translate short names to names readable for splitVs
	if varStr == "dfg_ql_srvy":
		varStr = "B_dmpa_dfg_srvy_ql"
	elif varStr == "afg_ql_srvy":
		varStr = "B_dmpa_afg_srvy_ql"
	elif varStr == "Nhplus_hpca_sitl":
		varStr = "Nhplus_hpca_srvy_sitl"
	elif varStr in ["R_gse","R_gsm","V_gse","V_gsm"]:
		varStr = "_".join([varStr,"mec","srvy","l2"])
		

	Vr = splitVs(varStr)
	mmsIdS = "mms"+probe
	Vr["dtype"] = None
	vdf_flag = False
	#pdb.set_trace()

	if Vr["inst"] == "mec":
		cdfname     = "_".join([mmsIdS,"mec",Vr["param"].lower(),Vr["cs"]])
		Vr["dtype"] = "epht89d"
	elif Vr["inst"] == "fsm":
		cdfname     = "_".join([mmsIdS,Vr["inst"],"b",Vr["cs"],Vr["tmmode"],Vr["lev"]])
		Vr["dtype"] = "8khz"

	elif Vr["inst"] in ["fgm","dfg","afg"]:
		if Vr["lev"] == "l2":
			cdfname = "_".join([mmsIdS,Vr["inst"],"b",Vr["cs"],Vr["tmmode"],Vr["lev"]])
		elif Vr["lev"] == "l2pre":
			cdfname = "_".join([mmsIdS,Vr["inst"],"b",Vr["cs"],Vr["tmmode"],Vr["lev"]])
		elif Vr["lev"] == "ql":
			cdfname = "_".join([mmsIdS,Vr["inst"],Vr["tmmode"],Vr["lev"]])
		else :
			raise InterruptedError("Should not be here")


	elif Vr["inst"] == "fpi":
		# get specie
		if Vr["param"][-1] == "i":
			sensor = "dis"
		elif Vr["param"][-1] == "e":
			sensor = "des"
		else :
			raise ValueError("invalid specie")

		if Vr["lev"] in ["l2","l2pre","l1b"]:
			if Vr["param"][:2] == "PD":
				vdf_flag = True
				Vr["dtype"] = sensor+"-dist"
			else :
				if len(Vr["param"]) > 4:
					if Vr["param"][:4] == "part":
						Vr["dtype"] = sensor+"-partmoms"
					else :
						Vr["dtype"] = sensor+"-moms"
				else :
					Vr["dtype"] = sensor+"-moms"
		elif Vr["lev"] == "ql":
			Vr["dtype"] = sensor
		else :
			raise InterruptedError("Should not be here")
		
		if Vr["param"] in ["PDi","PDe","PDerri","PDerre"]:
			if Vr["param"][:-1].lower() == "pd":
				cdfname = "_".join([mmsIdS,sensor,"dist",Vr["tmmode"]])
			elif len(Vr["param"]) == 6 and Vr["param"][:-1].lower() == "pderr":
				cdfname = "_".join([mmsIdS,sensor,"disterr",Vr["tmmode"]])
			else :
				raise InterruptedError("Should not be here")
		# Number density
		elif Vr["param"] in ["Ne","Ni"]:
			if Vr["lev"] in ["l2","l2pre"]:
				cdfname = "_".join([mmsIdS,sensor,"numberdensity",Vr["tmmode"]])
			elif Vr["lev"] == "l1b":
				cdfname = "_".join([mmsIdS,sensor,"numberdensity"])
			elif Vr["lev"] == "ql":
				cdfname = "_".join([mmsIdS,sensor,"numberdensity","fast"])
			elif Vr["lev"] == "sitl":
				cdfname = "_".join([mmsIdS,"fpi",sensor.upper(),"numberDensity"])
			else :
				raise InterruptedError("Should not be here")

		# Number density
		elif Vr["param"] in ["Nbge","Nbgi"]:
			if Vr["lev"] in ["l2","l2pre"]:
				cdfname = "_".join([mmsIdS,sensor,"numberdensity_bg",Vr["tmmode"]])
			elif Vr["lev"] == "l1b":
				cdfname = "_".join([mmsIdS,sensor,"numberdensity_bg"])
			elif Vr["lev"] == "ql":
				cdfname = "_".join([mmsIdS,sensor,"numberdensity_bg","fast"])
			else :
				raise InterruptedError("Should not be here")

		# Partial number density
		elif Vr["param"] in ["partNi","partNe"]:
			if Vr["lev"] == "l2":
				# only for l2 data
				cdfname = "_".join([mmsIdS,sensor,"numberdensity","part",Vr["tmmode"]])
			else :
				raise InterruptedError("Should not be here")

		# Energy flux omni
		elif Vr["param"] in ["Enfluxi","Enfluxe"]:
			if Vr["lev"] == "ql":
				cdfname = "_".join([mmsIdS,sensor,"energyspectr","omni","fast"])
			elif Vr["lev"] == "l2":
				cdfname = "_".join([mmsIdS,sensor,"energyspectr","omni",Vr["tmmode"]])
			else :
				raise InterruptedError("Should not be here")

		elif Vr["param"] in ["Enfluxbgi","Enfluxbge"]:
			if Vr["lev"] == "l2":
				cdfname = "_".join([mmsIdS,sensor,"spectr_bg",Vr["tmmode"]])
			else :
				raise InterruptedError("Should not be here")
		

		# Energies
		elif Vr["param"] in ["Energyi","Energye"]:
			if Vr["lev"] == "ql":
				cdfname = "_".join([mmsIdS,sensor,"energy","fast"])
			elif Vr["lev"] == "l2":
				cdfname = "_".join([mmsIdS,sensor,"energy",Vr["tmmode"]])
			else :
				raise InterruptedError("Should not be here")

		# Parallel and perpandiculat temperatures
		elif Vr["param"] in ["Tparai","Tparae", "Tperpi","Tperpe"]:
			tmpFAC = Vr["param"][1:5]
			if Vr["lev"] == "l2":
				cdfname = "_".join([mmsIdS,sensor,"temp"+tmpFAC,Vr["tmmode"]])
			else :
				raise InterruptedError("Should not be here")

		# Partial moments parallel and perpandiculat temperatures
		elif Vr["param"] in ["partTparai","partTparae","partTperpi","partTperpe"]:
			tmpFAC = Vr["param"][5:9]
			if Vr["lev"] == "l2":
				cdfname = "_".join([mmsIdS,sensor,"temp"+tmpFAC,"part",Vr["tmmode"]])
			else :
				raise InterruptedError("Should not be here")

		# Temperature and pressure tensors
		elif Vr["param"] in ["Ti","Te","Pi","Pe"]:
			if Vr["param"][0] == "T":
				momType = "temptensor" # temperature
			elif Vr["param"][0] == "P":
				momType = "prestensor" # pressure
			else :
				raise InterruptedError("Should not be here")

			cdfname = "_".join([mmsIdS,sensor,momType,Vr["cs"],Vr["tmmode"]])

		elif Vr["param"] in ["Pbgi","Pbge"]:
			momType = "pres_bg"
			cdfname = "_".join([mmsIdS,sensor,momType,Vr["tmmode"]])

		# Partial temperature and pressure tensors
		elif Vr["param"] in ["partTi","partTe","partPi","partPe"]:
			if Vr["param"][4] == "T":
				momType = "temptensor" # temperature
			elif Vr["param"][4] == "P":
				momType = "prestensor" # pressure
			else :
				raise InterruptedError("Should not be here")

			cdfname = "_".join([mmsIdS,sensor,momType,"part",Vr["cs"],Vr["tmmode"]])

		# spintone
		elif Vr["param"] in ["STi","STe"]:
			if Vr["lev"] in ["l2","l2pre"]:
				cdfname = "_".join([mmsIdS,sensor,"bulkv_spintone",Vr["cs"],Vr["tmmode"]])
			elif Vr["lev"] == "ql":
				cdfname = "_".join([mmsIdS,sensor,"bulkv_spintone",Vr["cs"],Vr["tmmode"]])
			else :
				raise InterruptedError("Should not be here")
		# Bulk velocities
		elif Vr["param"] in ["Vi","Ve"]:
			if Vr["lev"] in ["l2","l2pre"]:
				cdfname = "_".join([mmsIdS,sensor,"bulkv",Vr["cs"],Vr["tmmode"]])
			elif Vr["lev"] == "ql":
				cdfname = "_".join([mmsIdS,sensor,"bulkv",Vr["cs"],Vr["tmmode"]])
			else :
				raise InterruptedError("Should not be here")

		#Error bulk velocities
		elif Vr["param"] in ["errVi","errVe"]:
			if Vr["lev"] == "l2":
				cdfname = "_".join([mmsIdS,sensor,"bulkv","err",Vr["tmmode"]])
			else :
				raise InterruptedError("Only l2 partmoms available now")

		# Partial bulk velocities
		elif Vr["param"] in ["partVi","partVe"]:
			if Vr["lev"] == "l2":
				cdfname = "_".join([mmsIdS,sensor,"bulkv","part",Vr["cs"],Vr["tmmode"]])
			else :
				raise InterruptedError("Only l2 partmoms available now")

		elif Vr["param"] in ["PADlowene","PADmidene","PADhighene"]:
			if Vr["lev"] == "l2":
				cdfname = "_".join([mmsIdS,sensor,"pitchangdist",Vr["param"][3:-1],Vr["tmmode"]])
			else :
				raise InterruptedError("Only l2 partmoms available now")                
		else :
			raise InterruptedError("Should not be here")

	# Hot Plasma Composition Analyser
	elif Vr["inst"] == "hpca":
		param       = Vr["param"][0]
		Vr["dtype"] = "moments"
		# Get specie name
		ion = Vr["param"][1:]
		if ion[0] == "s":
			param   = param+ion[0]
			ion     = ion[1:]

		if not ion in ["hplus","heplus","heplusplus","oplus"]:
			raise ValueError("Unrecognized ion")

		# Number density
		if "N" in Vr["param"]:
			cdfname = "_".join([mmsIdS,"hpca",ion,"number_density"])
		# Bulk velocity
		elif "V" in Vr["param"]:
			cdfname = "_".join([mmsIdS,"hpca",ion,"ion_bulk_velocity"])
		# Scalar temperature
		elif "Ts" in Vr["param"]:
			cdfname = "_".join([mmsIdS,"hpca",ion,"scalar_temperature"])
		# Pressure tensor
		elif "P" in Vr["param"]:
			cdfname = "_".join([mmsIdS,"hpca",ion,"ion_pressure"])
		# Temperature tensor
		elif "T" in Vr["param"]:
			cdfname = "_".join([mmsIdS,"hpca",ion,"temperature_tensor"])
		elif "PSD" in Vr["param"]:
			cdfname = "_".join([mmsIdS,"hpca",ion,"phase_space_density"])
		else :
			raise ValueError("Unrecognized param")

		# Tensor (vector or matrix) add coordinate system to cdfname
		if Vr["to"] > 0:
			if Vr["cs"] == "gsm":
				cdfname = "_".join([cdfname,"GSM"])
			elif Vr["cs"] == "dbcs":
				pass
			else :
				raise ValueError("invalid CS")

	# Search Coil Magnetometer
	elif Vr["inst"] == "scm":
		if Vr["lev"] != "l2":
			raise InterruptedError("not implemented yet")

		cdfname     = "_".join([mmsIdS,"scm","acb",Vr["cs"],"scb",Vr["tmmode"],Vr["lev"]])
		Vr["dtype"] = "scb"
		
	elif Vr["inst"] == "dsp":
		if Vr["lev"] == "l2":
			if Vr["param"][0].lower() == "e":
				Vr["dtype"] = "{}psd".format(Vr["param"][0].lower())
				cdfname = "_".join([mmsIdS,"dsp",Vr["dtype"],"omni"])
			elif Vr["param"][0].lower() == "b":
				Vr["dtype"] = "{}psd".format(Vr["param"][0].lower())
				cdfname = "_".join([mmsIdS,"dsp",Vr["dtype"],"omni",Vr["tmmode"],Vr["lev"]])
			else : 
				raise ValueError("Should not be here")

	# Spin-plane Double Probe instrument 
	elif Vr["inst"] == "edp":
		if Vr["lev"] == "sitl":
			if Vr["param"] == "E":
				param       = "_".join(["dce_xyz",Vr["cs"]])
				Vr["dtype"] = "dce"

			elif Vr["param"] == "E2d":
				param       = "_".join(["dce_xyz",Vr["cs"]])
				Vr["dtype"] = "dce2d"

			elif Vr["param"] == "V":
				param       = "scpot"
				Vr["dtype"] = "scpot"

			cdfname = "_".join([mmsIdS,"edp",param,Vr["tmmode"],Vr["lev"]])

		elif Vr["lev"] == "ql":
			if Vr["param"] == "E":
				param       = "_".join(["dce_xyz",Vr["cs"]])
				Vr["dtype"] = "dce"

			elif Vr["param"] == "E2d":
				param       = "_".join(["dce_xyz",Vr["cs"]])
				Vr["dtype"] = "dce2d"

			cdfname = "_".join([mmsIdS,"edp",param])

		elif Vr["lev"] == "l1b":
			if Vr["param"] == "E":
				param       = "dce_sensor"
				Vr["dtype"] = "dce"

			elif Vr["param"] == "V":
				param       = "dcv_sensor"
				Vr["dtype"] = "dce"

			cdfname = "_".join([mmsIdS,"edp",param])

		elif Vr["lev"] == "l2a":
			Vr["dtype"] = "dce2d"

			if Vr["param"] == "Phase":
				param = "phase"

			elif Vr["param"] in ["Es12","Es34"]:
				param = "espin_p"+Vr["param"][2:4]

			elif Vr["param"] == "Adcoff":
				param = "adc_offset"

			elif Vr["param"] in ["Sdev12","Sdev34"]:
				param = "sdevfit_p"+Vr["param"][4:6]

			elif Vr["param"] == "E":
				param = "dce"

			cdfname = "_".join([mmsIdS,"edp",param,Vr["tmmode"],Vr["lev"]])

		else :
			if Vr["param"] == "E":
				param       = "dce_"+Vr["cs"]
				Vr["dtype"] = "dce"

			elif Vr["param"] == "Epar":
				param       = "dce_par_epar"
				Vr["dtype"] = "dce"

			elif Vr["param"] == "E2d":
				param       = "dce_"+Vr["cs"]
				Vr["dtype"] = "dce2d"

			elif Vr["param"] == "V":
				param       = "scpot"
				Vr["dtype"] = "scpot"

			elif Vr["param"] == "V6":
				Vr["dtype"] = "scpot"
				param = "dcv"

			cdfname = "_".join([mmsIdS,"edp",param,Vr["tmmode"],Vr["lev"]])

	elif Vr["inst"] == "epd-eis":
		if int(tint[0][2:4]) < 17:
			ver = "3"
		else :
			ver = "4"
		if Vr["lev"] == "l2":
			if "len" in Vr["param"]:
				Vr["dtype"] = "phxtof"
				if "proton" in Vr["param"].lower():
					if "flux" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","phxtof","proton","P4","flux",Vr["param"][-2:]])

						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"phxtof","proton","P4","flux",\
												Vr["param"][-2:]])

					elif "cps" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","phxtof","proton","P4","cps",Vr["param"][-2:]])

						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"phxtof","proton","P4","cps",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					elif "counts" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","phxtof","proton","P4","counts",Vr["param"][-2:]])
						
						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"phxtof","proton","P4","counts",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					else :
						raise InterruptedError("Should not be here")

				elif "oxygen" in Vr["param"].lower():
					if "flux" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","phxtof","oxygen","P4","flux",Vr["param"][-2:]])

						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"phxtof","oxygen","P4","flux",\
												Vr["param"][-2:]])

					elif "cps" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","phxtof","oxygen","P4","cps",Vr["param"][-2:]])

						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"phxtof","oxygen","P4","cps",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					elif "counts" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","phxtof","oxygen","P4","counts",Vr["param"][-2:]])
						
						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"phxtof","oxygen","P4","counts",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					else :
						raise InterruptedError("Should not be here")

				else :
					raise InterruptedError("Should not be here")
						
			elif Vr["param"][:3] == "hen":
				Vr["dtype"] = "extof"
				if "proton" in Vr["param"].lower():
					if "flux" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","extof","proton","P4","flux",Vr["param"][-2:]])

						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"extof","proton","P4","flux",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					elif "cps" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","extof","proton","P4","cps",Vr["param"][-2:]])

						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"extof","proton","P4","cps",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					elif "counts" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","extof","proton","P4","counts",Vr["param"][-2:]])
						
						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"extof","proton","P4","counts",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					else :
						raise InterruptedError("Should not be here")

				elif "oxygen" in Vr["param"].lower():
					if "flux" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","extof","oxygen","P4","flux",Vr["param"][-2:]])

						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"extof","oxygen","P4","flux",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					elif "cps" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","extof","oxygen","P4","cps",Vr["param"][-2:]])

						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"extof","oxygen","P4","cps",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					elif "counts" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","extof","oxygen","P4","counts",Vr["param"][-2:]])
						
						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"extof","oxygen","P4","counts",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					else :
						raise InterruptedError("Should not be here")

				elif "alpha" in Vr["param"].lower():
					if "flux" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","extof","alpha","P4","flux",Vr["param"][-2:]])

						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"extof","alpha","P4","flux",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					elif "cps" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","extof","alpha","P4","cps",Vr["param"][-2:]])

						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"extof","alpha","P4","cps",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					elif "counts" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","extof","alpha","P4","counts",Vr["param"][-2:]])
						
						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"extof","alpha","P4","counts",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					else :
						raise InterruptedError("Should not be here")
						
				elif "dump" in Vr["param"].lower():
					if "flux" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","extof","dump","P4","flux",Vr["param"][-2:]])

						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"dump","alpha","P4","flux",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					elif "cps" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","extof","dump","P4","cps",Vr["param"][-2:]])

						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"extof","dump","P4","cps",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					elif "counts" in Vr["param"].lower():
						if Vr["tmmode"] == "srvy":
							cdfname = "_".join([mmsIdS,"epd_eis","extof","dump","P4","counts",Vr["param"][-2:]])
						
						elif Vr["tmmode"] == "brst":
							cdfname = "_".join([mmsIdS,"epd_eis",Vr["tmmode"],"extof","dump","P4","counts",\
												Vr["param"][-2:]])

						else :
							raise InterruptedError("Should not be here")

					else :
						raise InterruptedError("Should not be here")

				else :
					raise InterruptedError("Should not be here")
					
			else :
				raise InterruptedError("Should not be here")
			
		else :
			raise ValueError("not implemented yet")

	elif Vr["inst"] == "feeps":
		sid = re.findall(r'\d+',Vr["param"])[0]
		if Vr["lev"] == "l2":
			if Vr["param"][-1] == "e":
				Vr["dtype"] = "electron"
				if "top" in Vr["param"].lower():
					if "flux" in Vr["param"].lower():
						cdfname = "_".join([mmsIdS,"epd_feeps",Vr["tmmode"],Vr["lev"],Vr["dtype"],"top",\
											"intensity","sensorid",sid])

					elif "cps" in Vr["param"].lower():
						cdfname = "_".join([mmsIdS,"epd_feeps",Vr["tmmode"],Vr["lev"],Vr["dtype"],"top",\
											"count_rate","sensorid",sid])

					elif "mask" in Vr["param"].lower():
						cdfname = "_".join([mmsIdS,"epd_feeps",Vr["tmmode"],Vr["lev"],Vr["dtype"],"top",\
											"sector_mask","sensorid",sid])

					else :
						raise InterruptedError("Should not be here")

				elif "bottom" in Vr["param"].lower():
					if "flux" in Vr["param"].lower():
						cdfname = "_".join([mmsIdS,"epd_feeps",Vr["tmmode"],Vr["lev"],Vr["dtype"],"bottom",\
											"intensity","sensorid",sid])

					elif "cps" in Vr["param"].lower():
						cdfname = "_".join([mmsIdS,"epd_feeps",Vr["tmmode"],Vr["lev"],Vr["dtype"],"bottom",\
											"count_rate","sensorid",sid])

					elif "mask" in Vr["param"].lower():
						cdfname = "_".join([mmsIdS,"epd_feeps",Vr["tmmode"],Vr["lev"],Vr["dtype"],"bottom",\
											"sector_mask","sensorid",sid])

					else :
						raise InterruptedError("Should not be here")

				else :
					raise InterruptedError("Should not be here")

			elif Vr["param"][-1] == "i":
				Vr["dtype"] = "ion"
				if "top" in Vr["param"].lower():
					if "flux" in Vr["param"].lower():
						cdfname = "_".join([mmsIdS,"epd_feeps",Vr["tmmode"],Vr["lev"],Vr["dtype"],"top",\
											"intensity","sensorid",sid])

					elif "cps" in Vr["param"].lower():
						cdfname = "_".join([mmsIdS,"epd_feeps",Vr["tmmode"],Vr["lev"],Vr["dtype"],"top",\
											"count_rate","sensorid",sid])

					elif "mask" in Vr["param"].lower():
						cdfname = "_".join([mmsIdS,"epd_feeps",Vr["tmmode"],Vr["lev"],Vr["dtype"],"top",\
											"sector_mask","sensorid",sid])

					else :
						raise InterruptedError("Should not be here")

				elif "bottom" in Vr["param"].lower():
					if "flux" in Vr["param"].lower():
						cdfname = "_".join([mmsIdS,"epd_feeps",Vr["tmmode"],Vr["lev"],Vr["dtype"],"bottom",\
											"intensity","sensorid",sid])

					elif "cps" in Vr["param"].lower():
						cdfname = "_".join([mmsIdS,"epd_feeps",Vr["tmmode"],Vr["lev"],Vr["dtype"],"bottom",\
											"count_rate","sensorid",sid])

					elif "mask" in Vr["param"].lower():
						cdfname = "_".join([mmsIdS,"epd_feeps",Vr["tmmode"],Vr["lev"],Vr["dtype"],"bottom",\
											"sector_mask","sensorid",sid])

					else :
						raise InterruptedError("Should not be here")

				else :
					raise InterruptedError("Should not be here")

		else :
			raise ValueError("not implemented yet")
	
	else :
		raise ValueError("not implemented yet")

	files = list_files(tint,probe,Vr)
	#files = list_files(data_path,tint,probe,Vr)
	if not files:
		raise ValueError("No files found. Make sure that the data_path is correct")

	if silent == False: print("Loading "+ cdfname+"...")


	for i, file in enumerate(files):
		if vdf_flag:
			temp = get_dist(file,cdfname,tint)
			if i == 0:
				out = temp
			else :
				out = dist_append(out,temp)
		else :
			temp = get_ts(file,cdfname,tint)
			if i == 0:
				out = temp
			else :
				out = ts_append(out,temp)

	if out.time.data.dtype == "float64":
				out.time.data = Time(1e-9*out.time.data,format="unix").datetime

	return out
		#files = pyRF.list_files(data_path,tint,probe,Vr["inst"],datatype,Vr["tmmode"],Vr["lev"])
#-----------------------------------------------------------------------------------------------------------------------