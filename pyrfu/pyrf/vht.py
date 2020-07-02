import numpy as np

from .resample import resample
from .e_vxb import e_vxb



def vht(e=None,b=None,flag=1):
	"""
	Estimate velocity of the De Hoffmann-Teller frame from the velocity estimate the electric field eht=-vhtxb

	Parameters :
		- e                 [xarray]                Electric field
		- b                 [xarray]                Magnetic field
		- flag              [int]                   Flag. If 2 assumed no Ez.

	Returns :
		- vht               [ndarray]               De Hoffmann Teller frame velocity [km/s]
		- vht               [xarray]                Electric field in the De Hoffmann frame             
		- dvht              [ndarray]               Error of De Hoffmann Teller frame
	
	"""
	
	if e is None:
		raise ValueError("eht requires at least 2 arguments")
		
	if b is None:
		raise ValueError("eht requires at least 2 arguments")
	
	nSamples = len(e)
	if nSamples != len(b):
		b = resample(b,e)

	bx = b[:,0].data
	by = b[:,1].data
	bz = b[:,2].data

	ex = b[:,0].data
	ey = b[:,1].data
	ez = b[:,2].data

	p       = np.zeros(6)
	p[0]    = np.sum(bx*bx)/nSamples # Bx*Bx
	p[1]    = np.sum(bx*by)/nSamples # Bx*By
	p[2]    = np.sum(bx*bz)/nSamples # Bx*Bz
	p[3]    = np.sum(by*by)/nSamples # By*By
	p[4]    = np.sum(by*bz)/nSamples # By*Bz
	p[5]    = np.sum(bz*bz)/nSamples # Bz*Bz


	if flag == 2:   # assume only Ex and Ey
		z       = 0       # put z component to 0 when using only Ex and Ey
		K       = np.array([[p[5],0,-p[2]],[0,p[5],-p[4]],[-p[2],-p[4],p[0]+p[3]]])
		comm    = "De Hoffmann-Teller frame is calculated using 2 components of E=(Ex,Ey,0)"
	else :
		K   = np.array([[p[3]+p[5],-p[1],-p[2]],[-p[1],p[0]+p[5],-p[4]],[-p[2],-p[4],p[0]+p[3]]])
		com = "De Hoffmann-Teller frame is calculated using all 3 components of E=(Ex,Ey,Ez)"

	ExB     = np.cross(e,b)
	indData = np.where(~np.isnan(ExB[:,0].data))[0]
	# revised by Wenya LI; 2015-11-21, wyli @ irfu
	tmp1    = ExB[indData]
	averExB = np.sum(tmp1,axis=0)/nSamples
	# averExB=sum(ExB(indData).data,1)/nSamples;
	# end revise.     
	VHT = np.linalg.solve(K,averExB.T)*1e3 # 9.12 in ISSI book

	vht     = VHT
	nvht    = vht/np.linalg.norm(vht)
	
	print(com)
	frmt = "V_HT ={:7.4f}*[{nvht[0]:7.4f}, {nvht[1]:7.4f}, {nvht[2]:7.4f}] "\
			+ "= [{vht[0]:7.4f}, {vht[1]:7.4f}, {vht[2]:7.4f}] km/s"
	print(frmt.format(np.linalg.norm(vht),nvht=nvht,vht=vht))


	#
	# Calculate the goodness of the Hofmann Teller frame
	#
	eht = e_vxb(vht,b);

	if flag == 2:
		ep              = e[indData]
		ehtp            = eht[indData]
		ep.data[:,2]    = 0
		ehtp.data[:,2]  = 0
	else :
		ep      = e[indData]
		ehtp    = eht[indData]

	deltaE = ep.data -ehtp.data

	p   = np.polyfit(ehtp.data.reshape([len(ehtp)*3]),ep.data.reshape([len(ep)*3]),1)
	cc  = np.corrcoef(ehtp.data.reshape([len(ehtp)*3]),ep.data.reshape([len(ep)*3]))

	print("slope = {p[0]:6.4f}, offs = {p[1]:6.4f}".format(p=p))
	print("cc = {:6.4f}".format(cc[0,1]))

	DVHT    = np.sum(np.sum(deltaE**2))/len(indData)
	S       = (DVHT/(2*len(indData)-3))/K
	dvht    = np.sqrt(np.diag(S))*1e3
	ndvht   = dvht/np.linalg.norm(dvht)
	frmt    = "\delta V_HT ={:7.4f}*[{ndvht[0]:7.4f}, {ndvht[1]:7.4f}, {ndvht[2]:7.4f}] "\
				+ "= [{dvht[0]:7.4f}, {dvht[1]:7.4f}, {dvht[2]:7.4f}] km/s"
	print(frmt.format(np.linalg.norm(dvht),ndvht=ndvht,dvht=dvht))

	return (vht,eht,dvht,p,cc)