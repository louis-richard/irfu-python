import seaborn as sns
from cycler import cycler
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import axes3d
from matplotlib.transforms import (Bbox, TransformedBbox, blended_transform_factory)
from mpl_toolkits.axes_grid1.inset_locator import (BboxPatch, BboxConnector, BboxConnectorPatch)

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib._color_data as mcd

plt.style.use("seaborn-whitegrid")
date_form = mdates.DateFormatter("%H:%M:%S")
sns.set_context("paper")
#plt.rc('text',usetex=True)
plt.rc('font',family='serif')
plt.rc('lines', linewidth=0.5)
color = ["k","b","r","g"]
plt.close("all")

def plot_config(R1,R2,R3,R4):
	"""
	Plot spacecraft configuaration with three 2d plots of the position in Re and one 3d plot of the relative position of
	the spacecraft:

	Parameters :
		- R1...R4           [xarray]                Position of the spacecraft 1...4

	Returns :
		- fig               [figure]
		- axs               [axes]

	"""

	R_E = constants.R_earth.value/1000
	r1 = np.mean(R1,0)
	r2 = np.mean(R2,0)
	r3 = np.mean(R3,0)
	r4 = np.mean(R4,0)
	R = np.vstack([r1,r2,r3,r4])
	r = np.mean(R,0)
	dR = R-np.tile(r,(4,1))
	
	
	fig = plt.figure(figsize=(9,9))
	gs0 = fig.add_gridspec(3, 3,hspace=0.3,left=0.1,right=0.9,bottom=0.1,top=0.9)

	gs00 = gs0[0,:].subgridspec(1, 3,wspace=0.35)
	gs10 = gs0[1:,:].subgridspec(1, 1,wspace=0.35)

	axs0 = fig.add_subplot(gs00[0])
	axs1 = fig.add_subplot(gs00[1])
	axs2 = fig.add_subplot(gs00[2])
	axs3 = fig.add_subplot(gs10[0], projection='3d')
	
	axs0.scatter(R[0,0]/R_E,R[0,2]/R_E,marker="s")
	axs0.scatter(R[1,0]/R_E,R[1,2]/R_E,marker="d")
	axs0.scatter(R[2,0]/R_E,R[2,2]/R_E,marker="o")
	axs0.scatter(R[3,0]/R_E,R[3,2]/R_E,marker="^")
	earth = plt.Circle((0, 0), 1, color='k', clip_on=False)
	axs0.add_artist(earth)
	axs0.set_xlim([20,-20])
	axs0.set_ylim([-20,20])
	axs0.set_aspect("equal")
	axs0.set_xlabel("$X$ [$R_e$]")
	axs0.set_ylabel("$Z$ [$R_e$]")
	axs0.set_title("X = {:2.1f} $R_E$".format(r[0]/R_E))
	
	axs1.scatter(R[0,1]/R_E,R[0,2]/R_E,marker="s")
	axs1.scatter(R[1,1]/R_E,R[1,2]/R_E,marker="d")
	axs1.scatter(R[2,1]/R_E,R[2,2]/R_E,marker="o")
	axs1.scatter(R[3,1]/R_E,R[3,2]/R_E,marker="^")
	earth = plt.Circle((0, 0), 1, color='k', clip_on=False)
	axs1.add_artist(earth)
	axs1.set_xlim([20,-20])
	axs1.set_ylim([-20,20])
	axs1.set_aspect("equal")
	axs1.set_xlabel("$Y$ [$R_e$]")
	axs1.set_ylabel("$Z$ [$R_e$]")
	axs1.set_title("Y = {:2.1f} $R_E$".format(r[1]/R_E))
	
	axs2.scatter(R[0,0]/R_E,R[0,1]/R_E,marker="s")
	axs2.scatter(R[1,0]/R_E,R[1,1]/R_E,marker="d")
	axs2.scatter(R[2,0]/R_E,R[2,1]/R_E,marker="o")
	axs2.scatter(R[3,0]/R_E,R[3,1]/R_E,marker="^")
	earth = plt.Circle((0, 0), 1, color='k', clip_on=False)
	axs2.add_artist(earth)
	axs2.set_xlim([20,-20])
	axs2.set_ylim([-20,20])
	axs2.set_aspect("equal")
	axs2.set_xlabel("$X$ [$R_e$]")
	axs2.set_ylabel("$Y$ [$R_e$]")
	axs2.set_title("Z = {:2.1f} $R_E$".format(r[2]/R_E))
	
	axs3.view_init(elev=13,azim=-20)
	axs3.scatter(dR[0,0],dR[0,1],dR[0,2],s=50,marker="s")
	axs3.scatter(dR[1,0],dR[1,1],dR[1,2],s=50,marker="d")
	axs3.scatter(dR[2,0],dR[2,1],dR[2,2],s=50,marker="o")
	axs3.scatter(dR[3,0],dR[3,1],dR[3,2],s=50,marker="^")
	
	axs3.plot([dR[0,0]]*2,[dR[0,1]]*2,color=color[0],marker="s", zdir='z',zs=-30)
	axs3.plot([dR[1,0]]*2,[dR[1,1]]*2,color=color[1],marker="d", zdir='z',zs=-30)
	axs3.plot([dR[2,0]]*2,[dR[2,1]]*2,color=color[2],marker="o", zdir='z',zs=-30)
	axs3.plot([dR[3,0]]*2,[dR[3,1]]*2,color=color[3],marker="^", zdir='z',zs=-30)
	
	axs3.plot([dR[0,0]]*2,[dR[0,2]]*2,color=color[0],marker="s", zdir='y',zs=-30)
	axs3.plot([dR[1,0]]*2,[dR[1,2]]*2,color=color[1],marker="d", zdir='y',zs=-30)
	axs3.plot([dR[2,0]]*2,[dR[2,2]]*2,color=color[2],marker="o", zdir='y',zs=-30)
	axs3.plot([dR[3,0]]*2,[dR[3,2]]*2,color=color[3],marker="^", zdir='y',zs=-30)
	
	axs3.plot([dR[0,1]]*2,[dR[0,2]]*2,color=color[0],marker="s", zdir='x',zs=-30)
	axs3.plot([dR[1,1]]*2,[dR[1,2]]*2,color=color[1],marker="d", zdir='x',zs=-30)
	axs3.plot([dR[2,1]]*2,[dR[2,2]]*2,color=color[2],marker="o", zdir='x',zs=-30)
	axs3.plot([dR[3,1]]*2,[dR[3,2]]*2,color=color[3],marker="^", zdir='x',zs=-30)
	
	
	axs3.plot([dR[0,0]]*2,[dR[0,1]]*2,[-30,dR[0,2]],'k--',linewidth=.5)
	axs3.plot([dR[1,0]]*2,[dR[1,1]]*2,[-30,dR[1,2]],'k--',linewidth=.5)
	axs3.plot([dR[2,0]]*2,[dR[2,1]]*2,[-30,dR[2,2]],'k--',linewidth=.5)
	axs3.plot([dR[3,0]]*2,[dR[3,1]]*2,[-30,dR[3,2]],'k--',linewidth=.5)
	
	axs3.plot([dR[0,0]]*2,[-30,dR[0,1]],[dR[0,2]]*2,'k--',linewidth=.5)
	axs3.plot([dR[1,0]]*2,[-30,dR[1,1]],[dR[1,2]]*2,'k--',linewidth=.5)
	axs3.plot([dR[2,0]]*2,[-30,dR[2,1]],[dR[2,2]]*2,'k--',linewidth=.5)
	axs3.plot([dR[3,0]]*2,[-30,dR[3,1]],[dR[3,2]]*2,'k--',linewidth=.5)
	
	axs3.plot([-30,dR[0,0]],[dR[0,1]]*2,[dR[0,2]]*2,'k--',linewidth=.5)
	axs3.plot([-30,dR[1,0]],[dR[1,1]]*2,[dR[1,2]]*2,'k--',linewidth=.5)
	axs3.plot([-30,dR[2,0]],[dR[2,1]]*2,[dR[2,2]]*2,'k--',linewidth=.5)
	axs3.plot([-30,dR[3,0]],[dR[3,1]]*2,[dR[3,2]]*2,'k--',linewidth=.5)
	
	axs3.plot(dR[[0,1],0],dR[[0,1],1],dR[[0,1],2],'k-')
	axs3.plot(dR[[1,2],0],dR[[1,2],1],dR[[1,2],2],'k-')
	axs3.plot(dR[[2,0],0],dR[[2,0],1],dR[[2,0],2],'k-')
	axs3.plot(dR[[0,3],0],dR[[0,3],1],dR[[0,3],2],'k-')
	axs3.plot(dR[[1,3],0],dR[[1,3],1],dR[[1,3],2],'k-')
	axs3.plot(dR[[2,3],0],dR[[2,3],1],dR[[2,3],2],'k-')
	
	
	axs3.set_xlim([-30,30])
	axs3.set_ylim([30,-30])
	axs3.set_zlim([-30,30])
	axs3.set_xlabel("$\\Delta X$ [km]")
	axs3.set_ylabel("$\\Delta Y$ [km]")
	axs3.set_zlabel("$\\Delta Z$ [km]")
		
	axs3.legend(["MMS1","MMS2","MMS3","MMS4"],frameon=False)

	axs = [axs0,axs1,axs2,axs3]
	
	return (fig, axs)