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


def connect_bbox(bbox1, bbox2,
				 loc1a, loc2a, loc1b, loc2b,
				 prop_lines, prop_patches=None):
	if prop_patches is None:
		prop_patches = {
			**prop_lines,
			"alpha": prop_lines.get("alpha", 1) * 0,
		}

	c1 = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a, **prop_lines)
	c1.set_clip_on(False)
	c2 = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b, **prop_lines)
	c2.set_clip_on(False)

	bbox_patch1 = BboxPatch(bbox1, **prop_patches)
	bbox_patch2 = BboxPatch(bbox2, **prop_patches)

	p = BboxConnectorPatch(bbox1, bbox2,
						   # loc1a=3, loc2a=2, loc1b=4, loc2b=1,
						   loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
						   **prop_patches)
	p.set_clip_on(False)

	return c1, c2, bbox_patch1, bbox_patch2, p


def zoom(ax1, ax2, **kwargs):
	"""
	ax1 : the main axes
	ax1 : the zoomed axes

	Similar to zoom_effect01.  The xmin & xmax will be taken from the
	ax1.viewLim.
	"""

	tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
	trans = blended_transform_factory(ax2.transData, tt)

	mybbox1 = ax1.bbox
	mybbox2 = TransformedBbox(ax1.viewLim, trans)

	prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}

	c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
		mybbox1, mybbox2,
		loc1a=2, loc2a=3, loc1b=1, loc2b=4,
		prop_lines=kwargs)

	ax1.add_patch(bbox_patch1)
	ax2.add_patch(bbox_patch2)
	ax2.add_patch(c1)
	ax2.add_patch(c2)
	ax2.add_patch(p)

	#return c1, c2, bbox_patch1, bbox_patch2, p
	return