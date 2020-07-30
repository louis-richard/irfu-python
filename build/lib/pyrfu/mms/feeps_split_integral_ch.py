import xarray as xr


def feeps_split_integral_ch(inp_dset):
	"""
    This function splits the last integral channel from the FEEPS spectra, 
    creating 2 new tplot variables:
       [original variable]_clean - spectra with the integral channel removed
       [original variable]_500keV_int - the integral channel that was removed
    
    Parameters:
        units_type: str
            instrument datatype, e.g., 'intensity'
        species: str
            'electron' or 'ion'
        probe: str
            probe #, e.g., '4' for MMS4
        suffix: str
            suffix of the loaded data
        data_rate: str
            instrument data rate, e.g., 'srvy' or 'brst'
        level: str
            data level
        sensor_eyes: dict
            Hash table containing the active sensor eyes
    Returns:
        List of tplot variables created.
    """

	outdict = {}
	outdict_500keV = {}

	for k in inp_dset:
		try :
			outdict[k] 			= inp_dset[k][:,:-1]
			outdict_500keV[k] 	= inp_dset[k][:,-1]
		except IndexError:
			pass

	out 		= xr.Dataset(outdict,attrs=inp_dset.attrs)
	out_500keV 	= xr.Dataset(outdict_500keV,attrs=inp_dset.attrs)

	#out.attrs["spin_sectors"] 		   = inp_dset.attrs["spin_sectors"]
	#out_500keV.attrs["spin_sectors"]   = inp_dset.attrs["spin_sectors"]

	return (out,out_500keV)