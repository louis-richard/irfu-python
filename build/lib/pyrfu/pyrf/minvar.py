import xarray as xr
import numpy as np



def minvar(inp=None, flag="mvar"):
	"""
	Compute the minimum variance frame

	Parameters :
		inp : DataArray
			Time series of the quantity to find minimum variance frame

	Options :
		flag : str
			Constrain

	Returns : 
		out : DataArray
			Time series of the input quantity in LMN coordinates

		l : array
			Eigenvalues l[0]>l[1]>l[2]

		V : array
			Eigenvectors LMN coordinates

	See also :
		new_xyz

	Example :
		>>> # Time interval
		>>> Tint = ["2019-09-14T07:54:00.000","2019-09-14T08:11:00.000"]
		>>> # Spacecraft index
		>>> ic = 1
		>>> # Load magnetic field
		>>> Bxyz = mms.get_data("B_gse_fgm_srvy_l2",Tint,ic)
		>>> # Compute MVA frame
		>>> Blmn, l, V = pyrf.minvar(Bxyz)

	"""

	inpdata = inp.data

	ooo=inpdata
	try:
		lx = inpdata.shape[0]
	except IndexError:
		lx = 1

	inp_m   = np.mean(inpdata,0)
	Mm2     = inp_m[[0,1,2,0,0,1]]*inp_m[[0,1,2,1,2,2]]
	Mm1     = np.mean(inpdata[:,[0,1,2,0,0,1]]*inpdata[:,[0,1,2,1,2,2]],0)

	if flag in ["mvar","<bn>=0"]:
		Mm  = Mm1-Mm2
		M   = np.array([Mm[[0,3,4]],Mm[[3,1,5]],Mm[[4,5,2]]])
	elif flag.lower()=="td":
		Mm  = Mm1
		M   = np.array([Mm[[0,3,4]],Mm[[3,1,5]],Mm[[4,5,2]]])

	[l,V]   = np.linalg.eig(M)
	idx     = l.argsort()[::-1] 
	l       = l[idx]
	V       = V[:,idx]
	V[:,2]  = np.cross(V[:,0],V[:,1])

	if flag.lower() == "<bn>=0":
		inp_mvar_mean = np.mean(np.sum(np.tile(inpdata,(3,1,1))\
									   *np.transpose(np.tile(V,(inpdata.shape[0],1,1)),(2,0,1)),1),1)

		a = np.sum(inp_mvar_mean**2)

		b = -(l[1]+l[2])*inp_mvar_mean[0]**2
		b -= (l[0]+l[2])*inp_mvar_mean[1]**2 
		b -= (l[0]+l[1])*inp_mvar_mean[2]**2

		c = l[1]*l[2]*inp_mvar_mean[0]**2 
		c += l[0]*l[2]*inp_mvar_mean[1]**2 
		c += l[0]*l[1]*inp_mvar_mean[2]**2

		r = np.roots([a,b,c])
		lmin=np.min(r)

		n = inp_mvar_mean/(l - lmin)
		nnorm = np.linalg.norm(n)
		n = n/nnorm

		n = np.matmul(V,n)

		bn          = np.sum(inpdata*np.tile(n,(inpdata.shape[0],1)),axis=1)
		inpdata_2   = inpdata - np.tile(bn,(3,1)).T*np.tile(n,(inpdata.shape[0],1))
		inpdata_2_m = np.mean(inpdata_2,0)
		Mm2         = inpdata_2_m[[0,1,2,0,0,1]]*inpdata_2_m[[0,1,2,1,2,2]]
		Mm1         = np.mean(inpdata_2[:,[0,1,2,0,0,1]]*inpdata_2[:,[0,1,2,1,2,2]],0)
		Mm          = Mm1-Mm2
		M           = np.array([Mm[[0,3,4]],Mm[[3,1,5]],Mm[[4,5,2]]])
		[l,V]       = np.linalg.eig(M)
		idx         = l.argsort()[::-1] 
		l           = l[idx]
		V           = V[:,idx]
		V[:,2]      = np.cross(V[:,0],V[:,1])
		l[2]        = lmin

	elif flag.lower() == "td":
		ln          = l[2]
		bn          = np.sum(inpdata*np.tile(V[:,2],(inpdata.shape[0],1)),axis=1)
		inpdata_2   = inpdata - np.tile(bn,(3,1)).T*np.tile(V[:,2],(inpdata.shape[0],1))
		inpdata_2_m = np.mean(inpdata_2,0)
		Mm2         = inpdata_2_m[[0,1,2,0,0,1]]*inpdata_2_m[[0,1,2,1,2,2]]
		Mm1         = np.mean(inpdata_2[:,[0,1,2,0,0,1]]*inpdata_2[:,[0,1,2,1,2,2]],0)
		Mm          = Mm1-Mm2
		M           = np.array([Mm[[0,3,4]],Mm[[3,1,5]],Mm[[4,5,2]]])
		[l,V]       = np.linalg.eig(M)
		idx         = l.argsort()[::-1] 
		l           = l[idx]
		V           = V[:,idx]
		V[:,2]      = np.cross(V[:,0],V[:,1])
		l[2]        = ln

	outdata = (V.T @ inpdata.T).T

	out = xr.DataArray(outdata,coords=inp.coords,dims=inp.dims)

	

	return(out,l,V)