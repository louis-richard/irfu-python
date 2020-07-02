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




def plot_spectr(ax=None,inp=None,yscale="",ylim=None,cscale="",clim=None,cmap="",cbar=True,**kwargs):
	"""
	Plot a spectrogram using pcolormesh. 

	Parameters :
		- ax                [axes]                  Target axis to plot. If None create a new figure
		- inp               [axes]                  Input 2D data to plot
		- yscale            [str]                   Y-axis flag. Default is "" (linear)
		- ylim              [list]                  Y-axis bounds. Default is None (autolim)
		- cscale            [str]                   C-axis flag. Default is "" (linear)
		- clim              [list]                  C-axis bounds. Default is None (autolim)
		- cmap              [str]                   Colormap. Default is jet
		- cbar              [bool]                  Flag for colorbar. Set to False to hide

	Returns :
		- fig               [figure]
		- axs               [axes]
		- caxs              [caxes]                 Only if cbar is True

	"""
	if ax == None: 
		fig, ax = plt.subplots(1)
	else :
		fig = plt.gcf()
	
	if cscale == "log":
		if clim != None and isinstance(clim,list):
			#inp.data[inp.data == 0] = 1e-15
			norm = colors.LogNorm(vmin=clim[0],vmax=clim[1])
			vmin = clim[0]
			vmax = clim[1]
		else :
			#inp.data[inp.data == 0] = 1e-15
			norm = colors.LogNorm()
			vmin = None
			vmax = None
	else :
		if clim != None and isinstance(clim,list):
			norm = None
			vmin = clim[0]
			vmax = clim[1]
		else :
			norm = None
			vmin = None
			vmax = None
	if not cmap: cmap = "jet"
	t = inp.coords[inp.dims[0]]
	y = inp.coords[inp.dims[1]]
	im = ax.pcolormesh(t,y,inp.data.T,norm=norm,cmap=cmap,vmin=vmin,vmax=vmax)

	if yscale == "log": ax.set_yscale("log")

	if cbar:
		if "pad" in kwargs:
			pad = kwargs["pad"]
		else :
			pad = 0.01

		pos = ax.get_position()
		cax = fig.add_axes([pos.x0+pos.width+pad,pos.y0,0.01,pos.height])
		fig.colorbar(mappable=im,cax=cax,ax=ax)
		return (ax, cax)
	else :
		return ax
