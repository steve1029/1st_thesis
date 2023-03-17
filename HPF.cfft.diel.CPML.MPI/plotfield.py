import numpy as np
import os, datetime, sys
from scipy.constants import c

class Graphtool(object):

	def __init__(self, Space, path):

		self.Space = Space
		savedir = path + 'graph/'
		self.savedir = savedir 

		if self.Space.MPIrank == 0 : 

			while (os.path.exists(path) == False):

				print("Directory you put does not exists")
				path = input()
				
				if os.path.exists(path) == True: break
				else: continue

			if os.path.exists(savedir) == False: os.mkdir(savedir)
			else: pass

	def plot2D3D(self, what, tstep, xidx=None, yidx=None, zidx=None, **kwargs):
		"""Plot 2D and 3D graph for a given field and position

		Parameters
		----------
		what : string
			field to plot.
		figsize : tuple
			size of the figure.

		Return
		----------
		None
		"""

		###################################################################################
		###################### Gather field data from all slave nodes #####################
		###################################################################################
		
		if   what == 'Ex':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Ex_re, root=0)
			self.gathered_fields_im = self.Space.MPIcomm.gather(self.Space.Ex_im, root=0)
		elif what == 'Ey':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Ey_re, root=0)
			self.gathered_fields_im = self.Space.MPIcomm.gather(self.Space.Ey_im, root=0)
		elif what == 'Ez':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Ez_re, root=0)
			self.gathered_fields_im = self.Space.MPIcomm.gather(self.Space.Ez_im, root=0)
		elif what == 'Hx':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Hx_re, root=0)
			self.gathered_fields_im = self.Space.MPIcomm.gather(self.Space.Hx_im, root=0)
		elif what == 'Hy':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Hy_re, root=0)
			self.gathered_fields_im = self.Space.MPIcomm.gather(self.Space.Hy_im, root=0)
		elif what == 'Hz':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Hz_re, root=0)
			self.gathered_fields_im = self.Space.MPIcomm.gather(self.Space.Hz_im, root=0)

		if self.Space.MPIrank == 0: 

			try:
				import matplotlib.pyplot as plt
				from mpl_toolkits.mplot3d import axes3d
				from mpl_toolkits.axes_grid1 import make_axes_locatable

			except ImportError as err:
				print("Please install matplotlib at rank 0")
				sys.exit()

			colordeep = .1
			stride	  = 1
			zlim	  = 1
			figsize   = (15, 15)
			cmap      = plt.cm.bwr
			lc		  = 'b'
			aspect    = 'auto'

			for key, value in kwargs.items():

				if   key == 'colordeep': colordeep = value
				elif key == 'stride'   : stride    = value
				elif key == 'zlim'     : zlim      = value
				elif key == 'figsize'  : figsize   = value
				elif key == 'cmap'     : cmap      = value
				elif key == 'lc'	   : lc		   = value
				elif key == 'aspect'   : aspect    = value

			if xidx != None : 
				assert type(xidx) == int
				yidx  = slice(None,None) # indices from beginning to end
				zidx  = slice(None,None)
				plane = 'yz'
				col = np.arange(self.Space.Ny)
				row = np.arange(self.Space.Nz)
				plane_to_plot = np.zeros((len(col),len(row)), dtype=self.Space.dtype)

			elif yidx != None :
				assert type(yidx) == int
				xidx  = slice(None,None)
				zidx  = slice(None,None)
				plane = 'xz'
				col = np.arange(self.Space.Nx)
				row = np.arange(self.Space.Nz)
				plane_to_plot = np.zeros((len(col), len(row)), dtype=self.Space.dtype)

			elif zidx != None :
				assert type(zidx) == int
				xidx  = slice(None,None)
				yidx  = slice(None,None)
				plane = 'xy'
				col = np.arange(self.Space.Nx)
				row = np.arange(self.Space.Ny)
				plane_to_plot = np.zeros((len(col),len(row)), dtype=self.Space.dtype)
		
			elif (xidx,yidx,zidx) == (None,None,None):
				raise ValueError("Plane is not defined. Please insert one of x,y or z index of the plane.")

			#####################################################################################
			######### Build up total field with the parts of the grid from slave nodes ##########
			#####################################################################################

			integrated_field_re = np.zeros((self.Space.grid), dtype=self.Space.dtype)
			integrated_field_im = np.zeros((self.Space.grid), dtype=self.Space.dtype)

			for MPIrank in range(self.Space.MPIsize):
				integrated_field_re[self.Space.myNx_slices[MPIrank],:,:] = self.gathered_fields_re[MPIrank]
				integrated_field_im[self.Space.myNx_slices[MPIrank],:,:] = self.gathered_fields_im[MPIrank]

				#if MPIrank == 1: print(MPIrank, self.gathered_fields_re[MPIrank][xidx,yidx,zidx])

			#print(self.Space.myNx_slices[MPIrank])
			#print(MPIrank)
			#print(self.gathered_fields_re[MPIrank])
			#assert self.gathered_fields_re[MPIrank].all() != 0.

			plane_to_plot_re = integrated_field_re[xidx, yidx, zidx].copy()
			plane_to_plot_im = integrated_field_im[xidx, yidx, zidx].copy()

			Row, Col = np.meshgrid(row, col)
			today    = datetime.date.today()

			fig  = plt.figure(figsize=figsize)
			ax11 = fig.add_subplot(2,2,1)
			ax12 = fig.add_subplot(2,2,2, projection='3d')
			ax21 = fig.add_subplot(2,2,3)
			ax22 = fig.add_subplot(2,2,4, projection='3d')

			if plane == 'yz':
				
				image11 = ax11.imshow(plane_to_plot_re.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
				#image21 = ax21.imshow(plane_to_plot_im.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
				image21 = ax21.imshow(plane_to_plot_im.T, cmap=cmap, aspect=aspect)
				ax12.plot_wireframe(Col, Row, plane_to_plot_re[Col, Row], color=lc, rstride=stride, cstride=stride)
				ax22.plot_wireframe(Col, Row, plane_to_plot_im[Col, Row], color=lc, rstride=stride, cstride=stride)

				divider11 = make_axes_locatable(ax11)
				divider21 = make_axes_locatable(ax21)

				cax11  = divider11.append_axes('right', size='5%', pad=0.1)
				cax21  = divider21.append_axes('right', size='5%', pad=0.1)
				cbar11 = fig.colorbar(image11, cax=cax11)
				cbar21 = fig.colorbar(image21, cax=cax21)

				ax11.invert_yaxis()
				ax21.invert_yaxis()
				#ax12.invert_yaxis()
				#ax22.invert_yaxis()

				ax11.set_xlabel('y')
				ax11.set_ylabel('z')
				ax12.set_xlabel('y')
				ax12.set_ylabel('z')
				ax21.set_xlabel('y')
				ax21.set_ylabel('z')
				ax22.set_xlabel('y')
				ax22.set_ylabel('z')

			elif plane == 'xy':

				image11 = ax11.imshow(plane_to_plot_re.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
				#image21 = ax21.imshow(plane_to_plot_im.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
				image21 = ax21.imshow(plane_to_plot_im.T, cmap=cmap, aspect=aspect)
				ax12.plot_wireframe(Col, Row, plane_to_plot_re[Col, Row], color=lc, rstride=stride, cstride=stride)
				ax22.plot_wireframe(Col, Row, plane_to_plot_im[Col, Row], color=lc, rstride=stride, cstride=stride)

				divider11 = make_axes_locatable(ax11)
				divider21 = make_axes_locatable(ax21)

				cax11  = divider11.append_axes('right', size='5%', pad=0.1)
				cax21  = divider21.append_axes('right', size='5%', pad=0.1)
				cbar11 = fig.colorbar(image11, cax=cax11)
				cbar21 = fig.colorbar(image21, cax=cax21)

				ax11.invert_yaxis()
				ax21.invert_yaxis()
				#ax12.invert_yaxis()
				#ax22.invert_yaxis()

				ax11.set_xlabel('x')
				ax11.set_ylabel('y')
				ax12.set_xlabel('x')
				ax12.set_ylabel('y')
				ax21.set_xlabel('y')
				ax21.set_ylabel('x')
				ax22.set_xlabel('x')
				ax22.set_ylabel('y')

			elif plane == 'xz':

				image11 = ax11.imshow(plane_to_plot_re.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
				#image21 = ax21.imshow(plane_to_plot_im.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
				image21 = ax21.imshow(plane_to_plot_im.T, cmap=cmap, aspect=aspect)
				ax12.plot_wireframe(Col, Row, plane_to_plot_re[Col, Row], color=lc, rstride=stride, cstride=stride)
				ax22.plot_wireframe(Col, Row, plane_to_plot_im[Col, Row], color=lc, rstride=stride, cstride=stride)

				divider11 = make_axes_locatable(ax11)
				divider21 = make_axes_locatable(ax21)

				cax11  = divider11.append_axes('right', size='5%', pad=0.1)
				cax21  = divider21.append_axes('right', size='5%', pad=0.1)
				cbar11 = fig.colorbar(image11, cax=cax11)
				cbar21 = fig.colorbar(image21, cax=cax21)

				#ax11.invert_yaxis()
				#ax21.invert_yaxis()
				ax12.invert_yaxis()
				ax22.invert_yaxis()

				ax11.set_xlabel('x')
				ax11.set_ylabel('z')
				ax12.set_xlabel('x')
				ax12.set_ylabel('z')
				ax21.set_xlabel('z')
				ax21.set_ylabel('x')
				ax22.set_xlabel('x')
				ax22.set_ylabel('z')

			ax11.set_title(r'$%s.real, 2D$' %what)
			ax12.set_title(r'$%s.real, 3D$' %what)

			ax21.set_title(r'$%s.imag, 2D$' %what)
			ax22.set_title(r'$%s.imag, 3D$' %what)

			ax12.set_zlim(-zlim,zlim)
			ax22.set_zlim(-zlim,zlim)
			ax12.set_zlabel('field')
			ax22.set_zlabel('field')

			foldername = 'plot2D3D/'
			save_dir   = self.savedir + foldername

			if os.path.exists(save_dir) == False: os.mkdir(save_dir)
			fig.savefig('%s%s_%s_%s.png' %(save_dir, str(today), what, tstep), format='png', bbox_inches='tight')
			plt.close('all')

	def plot2D(self, what, tstep, xidx=None, yidx=None, zidx=None, **kwargs):
		"""Plot 2D graph for a given field and position

		Parameters
		------------
		what : string
			field to plot.
		figsize : tuple
			size of the figure.

		Return
		------------
		None
		"""

		###################################################################################
		###################### Gather field data from all slave nodes #####################
		###################################################################################
		
		if   what == 'Ex':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Ex_re, root=0)
			self.gathered_fields_im = self.Space.MPIcomm.gather(self.Space.Ex_im, root=0)
		elif what == 'Ey':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Ey_re, root=0)
			self.gathered_fields_im = self.Space.MPIcomm.gather(self.Space.Ey_im, root=0)
		elif what == 'Ez':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Ez_re, root=0)
			self.gathered_fields_im = self.Space.MPIcomm.gather(self.Space.Ez_im, root=0)
		elif what == 'Hx':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Hx_re, root=0)
			self.gathered_fields_im = self.Space.MPIcomm.gather(self.Space.Hx_im, root=0)
		elif what == 'Hy':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Hy_re, root=0)
			self.gathered_fields_im = self.Space.MPIcomm.gather(self.Space.Hy_im, root=0)
		elif what == 'Hz':
			self.gathered_fields_re = self.Space.MPIcomm.gather(self.Space.Hz_re, root=0)
			self.gathered_fields_im = self.Space.MPIcomm.gather(self.Space.Hz_im, root=0)

		if self.Space.MPIrank == 0: 

			try:
				import matplotlib.pyplot as plt
				from mpl_toolkits.mplot3d import axes3d
				from mpl_toolkits.axes_grid1 import make_axes_locatable

			except ImportError as err:
				print("Please install matplotlib at rank 0")
				sys.exit()

			colordeep = .1
			stride	  = 1
			zlim	  = 1
			figsize   = (15, 15)
			cmap      = plt.cm.bwr
			lc        = 'b'
			aspect    = 'auto'

			for key, value in kwargs.items():

				if   key == 'colordeep': colordeep = value
				elif key == 'stride'   : stride    = value
				elif key == 'zlim'     : zlim      = value
				elif key == 'figsize'  : figsize   = value
				elif key == 'cmap'     : cmap      = value
				elif key == 'lc'	   : lc		   = value
				elif key == 'aspect'   : aspect    = value

			if xidx != None : 
				assert type(xidx) == int
				yidx  = slice(None,None) # indices from beginning to end
				zidx  = slice(None,None)
				plane = 'yz'
				col   = np.arange(self.Space.Ny)
				row   = np.arange(self.Space.Nz)
				plane_to_plot = np.zeros((len(col),len(row)), dtype=self.Space.dtype)

			elif yidx != None :
				assert type(yidx) == int
				xidx  = slice(None,None)
				zidx  = slice(None,None)
				plane = 'xz'
				col   = np.arange(self.Space.Nx)
				row   = np.arange(self.Space.Nz)
				plane_to_plot = np.zeros((len(col), len(row)), dtype=self.Space.dtype)

			elif zidx != None :
				assert type(zidx) == int
				xidx  = slice(None,None)
				yidx  = slice(None,None)
				plane = 'xy'
				col   = np.arange(self.Space.Nx)
				row   = np.arange(self.Space.Ny)
				plane_to_plot = np.zeros((len(col),len(row)), dtype=self.Space.dtype)
		
			elif (xidx,yidx,zidx) == (None,None,None):
				raise ValueError("Plane is not defined. Please insert one of x,y or z index of the plane.")

			#####################################################################################
			######### Build up total field with the parts of the grid from slave nodes ##########
			#####################################################################################

			integrated_field_re = np.zeros((self.Space.grid), dtype=self.Space.dtype)
			integrated_field_im = np.zeros((self.Space.grid), dtype=self.Space.dtype)

			for MPIrank in range(self.Space.MPIsize):
				integrated_field_re[self.Space.myNx_slices[MPIrank],:,:] = self.gathered_fields_re[MPIrank]
				integrated_field_im[self.Space.myNx_slices[MPIrank],:,:] = self.gathered_fields_im[MPIrank]

				#if MPIrank == 1: print(MPIrank, self.gathered_fields_re[MPIrank][xidx,yidx,zidx])

			plane_to_plot_re = integrated_field_re[xidx, yidx, zidx].copy()
			plane_to_plot_im = integrated_field_im[xidx, yidx, zidx].copy()

			Row, Col = np.meshgrid(row, col)
			today    = datetime.date.today()

			fig  = plt.figure(figsize=figsize)
			ax11 = fig.add_subplot(2,1,1)
			ax12 = fig.add_subplot(2,1,2)

			fig_re = plt.figure(figsize=figsize)
			fig_im = plt.figure(figsize=figsize)

			ax_re = fig_re.add_subplot(1,1,1)
			ax_im = fig_im.add_subplot(1,1,1)

			if plane == 'yz':

				Row = Row[::-1]
				im = ax1.imshow(plane_to_plot.T[:,:].real, vmax=colordeep, vmin=-colordeep, cmap=cmap)
				ax2.plot_wireframe(Row,Col,plane_to_plot[Col,Row].real, \
									 color=lc, rstride=stride, cstride=stride)
				divider = make_axes_locatable(ax1)
				cax = divider.append_axes('right', size='5%', pad=0.1)
				cbar = fig.colorbar(im, cax=cax)

				ax1.set_xlabel('z')
				ax1.set_ylabel('y')
				ax2.set_xlabel('z')
				ax2.set_ylabel('y')

			elif plane == 'xy':

				im = ax1.imshow(plane_to_plot.T[Col,Row[::-1]].real, vmax=colordeep, vmin=-colordeep, cmap=cmap)
				ax2.plot_wireframe(Col,Row,plane_to_plot[Col,Rol].real, color=lc, rstride=stride, cstride=stride)
				divider = make_axes_locatable(ax1)
				cax = divider.append_axes('right', size='5%', pad=0.1)
				cbar = fig.colorbar(im, cax=cax)

				ax1.set_xlabel('x')
				ax1.set_ylabel('y')
				ax2.set_xlabel('x')
				ax2.set_ylabel('y')

			elif plane == 'xz':

				image11 = ax11.imshow(plane_to_plot_re.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
				image12 = ax12.imshow(plane_to_plot_im.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)

				image_re = ax_re.imshow(plane_to_plot_re.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
				image_im = ax_im.imshow(plane_to_plot_im.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)

				divider11 = make_axes_locatable(ax11)
				divider12 = make_axes_locatable(ax12)

				divider_re = make_axes_locatable(ax_re)
				divider_im = make_axes_locatable(ax_im)

				cax11  = divider11.append_axes('right', size='5%', pad=0.1)
				cax12  = divider12.append_axes('right', size='5%', pad=0.1)

				cax_re  = divider_re.append_axes('right', size='5%', pad=0.1)
				cax_im  = divider_im.append_axes('right', size='5%', pad=0.1)

				cbar11 = fig.colorbar(image11, cax=cax11)
				cbar12 = fig.colorbar(image12, cax=cax12)

				cbar_re = fig_re.colorbar(image_re, cax=cax_re)
				cbar_im = fig_im.colorbar(image_im, cax=cax_im)

				ax11.set_xlabel('x')
				ax11.set_ylabel('z')
				ax12.set_xlabel('x')
				ax12.set_ylabel('z')

				ax_re.set_xlabel('x')
				ax_re.set_ylabel('z')
				ax_im.set_xlabel('x')
				ax_im.set_ylabel('z')

			ax11.set_title(r'$%s.real, 2D$' %what)
			ax12.set_title(r'$%s.imag, 2D$' %what)

			ax_re.set_title(r'$%s.real, 2D$' %what)
			ax_im.set_title(r'$%s.imag, 2D$' %what)

			foldername = 'plot2D/'
			save_dir   = self.savedir + foldername

			if os.path.exists(save_dir) == False: os.mkdir(save_dir)

			fig.savefig   ('%s%s_%s_%s.png'    %(save_dir, str(today), what, tstep), format='png', bbox_inches='tight')
			fig_re.savefig('%s%s_%s_re_%s.png' %(save_dir, str(today), what, tstep), format='png', bbox_inches='tight')
			fig_im.savefig('%s%s_%s_im_%s.png' %(save_dir, str(today), what, tstep), format='png', bbox_inches='tight')

			plt.close('all')
