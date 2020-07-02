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


def plot_line(ax=None,inp=None,color="",yscale="",ylim=None):
	if ax == None: 
		fig, ax = plt.subplots(1)
	if len(inp.shape) == 3:
		data = np.reshape(inp.data,(inp.shape[0],inp.shape[1]*inp.shape[2]))
	else :
		data = inp.data
	time = inp.time
	ax.plot(time,data,color)
	date_form = mdates.DateFormatter("%H:%M:%S")
	ax.xaxis.set_major_formatter(date_form)
	ax.grid(which="major",linestyle="-",linewidth="0.5",c="0.5")
	#ax.grid(which="minor",linestyle="-",linewidth="0.25")
	return 