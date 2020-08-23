import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt


from ..pyrf import histogram2d
from . import plot_spectr







def pl_scatter_matrix(inp1=None,inp2=None, m="+", pdf = False, cmap="jet"):
	"""
	Produces a scatter plot of each components of field inp1 with respect to every component of field inp2. If pdf is 
	set to True, the scatter plot becomes a 2d histogram.

	Parameters :
		- inp1              [xarray]                First time series (x-axis)
		- inp2              [xarray]                Second time series (y-axis)
		- m                 [str]                   Marker type (optionnal). Default is "+". Not used if pdf is True
		- pdf               [bool]                  Flag to plot the 2d histogram. If False (default) the figure is a 
													scatter plot. If True the figure is a 2d histogram
		- cmap              [str]                   Colormap. Default : "jet"

	Returns :
		- fig               [figure]
		- axs               [axes]
		- caxs              [caxes]                 Only if pdf is True

	"""
	if inp1 is None:
		raise ValueError("pl_scatter_matrix requires at least one argument")
	
	if inp2 is None:
		inp2 = inp1
		warnings.warn("inp2 is empty assuming that inp2=inp1",UserWarning)
	
	if not isinstance(inp1,xr.DataArray) or not isinstance(inp2,xr.DataArray):
		raise TypeError("Inputs must be DataArrays")
	
	if not pdf:
		fig, axs = plt.subplots(3,3,sharex=True,sharey=True,figsize=(16,9))
		fig.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,hspace=0.05,wspace=0.05)
		for i in range(3):
			for j in range(3):
				axs[j,i].scatter(inp1[:,i].data,inp2[:,j].data,marker="+")
		return (fig, axs)
	else :
		fig, axs = plt.subplots(3,3,sharex=True,sharey=True,figsize=(16,9))
		fig.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,hspace=0.05,wspace=0.3)
		H = [[None]*3]*3
		caxs = [[None]*3]*3
		for i in range(3):
			for j in range(3):
				H[j][i] = histogram2d(inp1[:,i],inp2[:,j])
				axs[j,i], caxs[j][i] = plot_spectr(axs[j,i],H[j][i],cmap=cmap,cscale="log")
				axs[j,i].grid()
		return (fig, axs, caxs)
#-----------------------------------------------------------------------------------------------------------------------
