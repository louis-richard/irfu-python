import os

def splitVs(varStr=""):
	"""
	Parse the variable keys

	Paramters :
		- varStr            [str]                   Input key of variable

	Returns :
		- out               [dict]                  Dictionary containing : 
														- ["param"]     Variable key
														- ["to"]        Tensor order
														- ["cs"]        Coordinate system
														- ["inst"]      Instrument
														- ["tmmode"]    Time mode
														- ["lev"]       Level of data
	"""
	
	if not varStr:
		raise ValueError("splitVs requires at least one argument")
	
	if not isinstance(varStr,str):
		raise TypeError("varStr must be a string")
		
	tk = varStr.split("_")
	nTk = len(tk)
	if nTk < 3 or nTk > 5:
		raise ValueError("invalig STRING format")

	hpcaParamsScal  = ["Nhplus","Nheplus","Nheplusplus","Noplus","Tshplus",\
						"Tsheplus","Tsheplusplus","Tsoplus","Phase","Adcoff"]

	hpcaParamsTens  = ["Vhplus","Vheplus","Vheplusplus","Voplus","Phplus",\
						"Pheplus","Pheplusplus","Poplus","Thplus","Theplus",\
						"Theplusplus","Toplus"]

	eisParamsScal   = ["lenFluxprotont0","lenFluxprotont1","lenFluxprotont2",\
						"lenFluxprotont3","lenFluxprotont4","lenFluxprotont5",\
						"henFluxprotont0","henFluxprotont1","henFluxprotont2",\
						"henFluxprotont3","henFluxprotont4","henFluxprotont5",\
						"lenCPSprotont0","lenCPSprotont1","lenCPSprotont2",\
						"lenCPSprotont3","lenCPSprotont4","lenCPSprotont5",\
						"henCPSprotont0","henCPSprotont1","henCPSprotont2",\
						"henCPSprotont3","henCPSprotont4","henCPSprotont5",\
						"lenCountsprotont0","lenCountsprotont1","lenCountsprotont2",\
						"lenCountsprotont3","lenCountsprotont4","lenCountsprotont5",\
						"henCountsprotont0","henCountsprotont1","henCountsprotont2",\
						"henCountsprotont3","henCountsprotont4","henCountsprotont5",\
						"lenFluxoxygent0","lenFluxoxygent1","lenFluxoxygent2",\
						"lenFluxoxygent3","lenFluxoxygent4","lenFluxoxygent5",\
						"henFluxoxygent0","henFluxoxygent1","henFluxoxygent2",\
						"henFluxoxygent3","henFluxoxygent4","henFluxoxygent5",\
						"lenCPSoxygent0","lenCPSoxygent1","lenCPSoxygent2",\
						"lenCPSoxygent3","lenCPSoxygent4","lenCPSoxygent5",\
						"henCPSoxygent0","henCPSoxygent1","henCPSoxygent2",\
						"henCPSoxygent3","henCPSoxygent4","henCPSoxygent5",\
						"lenCountsoxygent0","lenCountsoxygent1","lenCountsoxygent2",\
						"lenCountsoxygent3","lenCountsoxygent4","lenCountsoxygent5",\
						"henCountsoxygent0","henCountsoxygent1","henCountsoxygent2",\
						"henCountsoxygent3","henCountsoxygent4","henCountsoxygent5",\
						"henFluxdumpt0","henFluxdumpt1","henFluxdumpt2",\
						"henFluxdumpt3","henFluxdumpt4","henFluxdumpt5",\
						"henCPSdumpt0","henCPSdumpt1","henCPSdumpt2",\
						"henCPSdumpt3","henCPSdumpt4","henCPSdumpt5",\
						"henCountsdumpt0","henCountsdumpt1","henCountsdumpt2",\
						"henCountsdumpt3","henCountsdumpt4","henCountsdumpt5",\
						"henFluxalphat0","henFluxalphat1","henFluxalphat2",\
						"henFluxalphat3","henFluxalphat4","henFluxalphat5",\
						"henCPSalphat0","henCPSalphat1","henCPSalphat2",\
						"henCPSalphat3","henCPSalphat4","henCPSalphat5",\
						"henCountsalphat0","henCountsalphat1","henCountsalphat2",\
						"henCountsalphat3","henCountsalphat4","henCountsalphat5"]

	feepsParamsScal = ["Fluxtop1e","Fluxtop2e","Fluxtop3e","Fluxtop4e",\
					"Fluxtop5e","Fluxtop6i","Fluxtop7i","Fluxtop8i",\
					"Fluxtop9e","Fluxtop10e","Fluxtop11e","Fluxtop12e",\
					"Fluxbottom1e","Fluxbottom2e","Fluxbottom3e","Fluxbottom4e",\
					"Fluxbottom5e","Fluxbottom6i","Fluxbottom7i","Fluxbottom8i",\
					"Fluxbottom9e","Fluxbottom10e","Fluxbottom11e","Fluxbottom12e",\
					"CPStop1e","CPStop2e","CPStop3e","CPStop4e",\
					"CPStop5e","CPStop6i","CPStop7i","CPStop8i",\
					"CPStop9e","CPStop10e","CPStop11e","CPStop12e",\
					"CPSbottom1e","CPSbottom2e","CPSbottom3e","CPSbottom4e",\
					"CPSbottom5e","CPSbottom6i","CPSbottom7i","CPSbottom8i",\
					"CPSbottom9e","CPSbottom10e","CPSbottom11e","CPSbottom12e",\
					"Masktop1e","Masktop2e","Masktop3e","Masktop4e",\
					"Masktop5e","Masktop6i","Masktop7i","Masktop8i",\
					"Masktop9e","Masktop10e","Masktop11e","Masktop12e",\
					"Maskbottom1e","Maskbottom2e","Maskbottom3e","Maskbottom4e",\
					"Maskbottom5e","Maskbottom6i","Maskbottom7i","Maskbottom8i",\
					"Maskbottom9e","Maskbottom10e","Maskbottom11e","Maskbottom12e"]

	param = tk[0]
	
	if param in ["Ni","Nbgi","Pbgi","partNi","Ne","Pbge","Nbge","partNe","Nhplus","Tsi","Tperpi","Tparai",\
				 "partTperpi","partTparai","Tse","Tperpe","Tparae","partTperpe",\
				 "partTparae","PDe","PDi","PDerre","PDerri","V","V6","Enfluxi","Enfluxbgi",\
				 "Enfluxe","Enfluxbge","Energyi","Energye","Epar","Sdev12","Sdev34","Flux-amb-pm2",\
				 "PADlowene","PADmidene","PADhighene","Bpsd","Epsd"]:
		tensorOrder = 0
	elif param in ["R","STi","Vi","errVi","partVi","STe","Ve","errVe","partVe","B","E","E2d","Es12","Es34"]:
		tensorOrder = 1
	elif param in ["Pi","partPi","Pe","partPe","Ti","partTi","Te","partTe"]:
		tensorOrder = 2
	elif param in hpcaParamsScal:
		tensorOrder = 0
	elif param in hpcaParamsTens:
		tensorOrder = 1
	elif param in eisParamsScal:
		tensorOrder =0
	elif param in feepsParamsScal:
		tensorOrder =0
	else :
		raise ValueError("invalid PARAM : {}".format(param))

	coordinateSystem = []
	idx = 0

	if tensorOrder > 0:
		coordinateSystem = tk[idx+1]
		idx             += 1

		if not coordinateSystem in ["gse","gsm","dsl","dbcs","dmpa","ssc","bcs","par"]:
			raise ValueError("invalid COORDINATE_SYS")

	instrument = tk[idx+1]
	idx += 1
	
	if not instrument in ["mec","fpi","edp","edi","hpca","fgm","dfg","afg","scm","fsm","epd-eis","feeps","dsp"]:
		raise ValueError("invalid INSTRUMENT")

	tmMode = tk[idx+1]
	idx += 1

	if not tmMode in ["brst","fast","slow","srvy"]:
		tmMode  = "fast"
		idx     -= 1
		warnings.warn("assuming TM_MODE = FAST",UserWarning)

	if len(tk) == idx+1:
		dataLevel = "l2" # default
	else :
		dataLevel = tk[idx+1]

		if not dataLevel in ["ql","sitl","l1b","l2a","l2pre","l2","l3"]:
			raise ValueError("invalid DATA_LEVEL level")

	res = { "param"     : param             ,\
			"to"        : tensorOrder       ,\
			"cs"        : coordinateSystem  ,\
			"inst"      : instrument        ,\
			"tmmode"    : tmMode            ,\
			"lev"       : dataLevel         ,\
			}
	

	return res