import numpy as np
import seaborn as sns
from cycler import cycler
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
date_form = mdates.ConciseDateFormatter(locator)
plt.style.use("seaborn-whitegrid")
sns.set_context("paper")

color = ["k", "b", "r", "g"]
default_cycler = cycler(color=color)
plt.rc('axes', prop_cycle=default_cycler)
plt.rc('lines', linewidth=1)

"""
date_form = mdates.DateFormatter("%H:%M:%S")
plt.rc('text',usetex=True)
plt.rc('font',family='serif')
plt.close("all")
"""


def plot_line(ax=None, inp=None, color="", yscale="", ylim=None):
	if ax is None:
		fig, ax = plt.subplots(1)
	if len(inp.shape) == 3:
		data = np.reshape(inp.data, (inp.shape[0], inp.shape[1] * inp.shape[2]))
	else:
		data = inp.data
	time = inp.time
	ax.plot(time, data, color)
	date_form = mdates.DateFormatter("%H:%M:%S")
	ax.xaxis.set_major_formatter(date_form)
	ax.grid(which="major", linestyle="-", linewidth="0.5", c="0.5")
	# ax.grid(which="minor",linestyle="-",linewidth="0.25")
	return
