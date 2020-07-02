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


def colorbar(im,ax,pad=0.01):
	"""
	Add colorbar to ax corresponding to im
	
	Parameters :
		- im
		- ax [axis] axis of plot
	"""
	pos = ax.get_position()
	fig = plt.gcf()
	cax = fig.add_axes([pos.x0+pos.width+pad,pos.y0,pad,pos.height])
	fig.colorbar(im,cax)
	return cax
