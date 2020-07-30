import seaborn as sns
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


plt.style.use("seaborn-whitegrid")
#date_form = mdates.DateFormatter("%H:%M:%S")
sns.set_context("paper")
#plt.rc('text',usetex=True)
#plt.rc('font',family='serif')
plt.rc('lines', linewidth=1)




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
