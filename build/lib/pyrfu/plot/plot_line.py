#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_line.py

@author : Louis RICHARD
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from cycler import cycler
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


def plot_line(ax=None, inp=None, c=""):
	if ax is None:
		fig, ax = plt.subplots(1)
	if len(inp.shape) == 3:
		data = np.reshape(inp.data, (inp.shape[0], inp.shape[1] * inp.shape[2]))
	else:
		data = inp.data

	time = inp.time
	ax.plot(time, data, c)
	date_form = mdates.DateFormatter("%H:%M:%S")
	ax.xaxis.set_major_formatter(date_form)
	ax.grid(which="major", linestyle="-", linewidth="0.5", c="0.5")
	# ax.grid(which="minor",linestyle="-",linewidth="0.25")

	return
