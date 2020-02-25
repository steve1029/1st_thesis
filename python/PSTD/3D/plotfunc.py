import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import axes3d
from scipy.constants import c, mu_0, epsilon_0
import datetime
import numpy as np

def plot2D3D(field,directory,**kwargs):
	"""Plot 2D and 3D field graph in one figure.
		Field array must be 2D. If field array is 3D,
		location of one of the axes must be specified.

	PARAMETERS
	--------------
	field : ndarray 
		2D numpy array to plot
	colordeep : float
		colordeep parameter of ax.imshow in 2D graph
	stride : int
		stride parameter of ax.plot_wireframe in 3D graph
	zlim : float
		zlim of 3D graph

	RETURN
	------------
	figure object
	"""
	step = ''
	colordeep = 1.
	stride = 1
	zlim = None

	for key, value in kwargs.items():
		if key == 'colordeep' : colordeep = value
		elif key == 'stride' : stride = value
		elif key == 'zlim' : zlim = value
		elif key == 'step' : step = value

	y = np.arange(field.shape[0])
	z = np.arange(field.shape[1])
	Z, Y = np.meshgrid(y,z)
	today = datetime.date.today()

	fig = plt.figure(figsize=(21,9))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122, projection='3d')

	im = ax1.imshow(field[:,:].real, vmax=colordeep, vmin=-colordeep, cmap=plt.cm.bwr)
	ax1.set_title('2D')
	ax1.set_xlabel('z')
	ax2.set_ylabel('y')
	divider = make_axes_locatable(ax1)
	cax = divider.append_axes('right', size='5%', pad=0.1)
	cbar = fig.colorbar(im,cax=cax)
	#ylabels = ax1.get_yticks().tolist()
	#ax1.set_yticklabels(ylabels[::-1])

	Y = Y[::-1]

	ax2.plot_wireframe(Y,Z,field[Z,Y].real,color='b',rstride=stride, cstride=stride)
	ax2.set_title('3D')
	ax2.set_xlabel('z')
	ax2.set_ylabel('y')
	if zlim != None : ax2.set_zlim(-zlim,zlim)
	fig.savefig('%s%s_%s.png' %(directory,str(today),step))
	plt.close('all')

