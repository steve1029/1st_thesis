import numpy as np
import os, datetime, sys

class Graphtool(object):

	def __init__(self, Space, path):

		self.Space = Space
		savedir = path + 'graph/'
		self.directory = savedir 

		if self.Space.MPIrank == 0 : 

			while (os.path.exists(path) == False):

				print("Directory you put does not exists")
				path = input()
				
				if os.path.exists(path) == True: break
				else: continue

			if os.path.exists(savedir) == False: os.mkdir(savedir)
			else: pass

	def plot2D3D(self, what, tstep, xidx=None, yidx=None, zidx=None, **kwargs):
		"""Plot 2D and 3D graph for given field and position

		Parameters
		------------
		what : string
			field to plot.
		figsize : tuple
			size of the figure.

		Return
		------
		None
		"""

		###################################################################################
		###################### Gather field data from all slave nodes #####################
		###################################################################################
		
		if   what == 'Ex':
			self.gathered_fields_re = self.Space.comm.gather(self.Space.Ex_re, root=0)
			self.gathered_fields_im = self.Space.comm.gather(self.Space.Ex_im, root=0)
		elif what == 'Ey':
			self.gathered_fields_re = self.Space.comm.gather(self.Space.Ey_re, root=0)
			self.gathered_fields_im = self.Space.comm.gather(self.Space.Ey_im, root=0)
		elif what == 'Ez':
			self.gathered_fields_re = self.Space.comm.gather(self.Space.Ez_re, root=0)
			self.gathered_fields_im = self.Space.comm.gather(self.Space.Ez_im, root=0)
		elif what == 'Hx':
			self.gathered_fields_re = self.Space.comm.gather(self.Space.Hx_re, root=0)
			self.gathered_fields_im = self.Space.comm.gather(self.Space.Hx_im, root=0)
		elif what == 'Hy':
			self.gathered_fields_re = self.Space.comm.gather(self.Space.Hy_re, root=0)
			self.gathered_fields_im = self.Space.comm.gather(self.Space.Hy_im, root=0)
		elif what == 'Hz':
			self.gathered_fields_re = self.Space.comm.gather(self.Space.Hz_re, root=0)
			self.gathered_fields_im = self.Space.comm.gather(self.Space.Hz_im, root=0)

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
			lc = 'b'

			for key, value in kwargs.items():

				if   key == 'colordeep': colordeep = value
				elif key == 'stride'   : stride    = value
				elif key == 'zlim'     : zlim      = value
				elif key == 'figsize'  : figsize   = value
				elif key == 'cmap'     : cmap      = value
				elif key == 'lc': lc = value

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

				image11 = ax11.imshow(plane_to_plot_re.T, vmax=colordeep, vmin=-colordeep, cmap=cmap)
				image21 = ax21.imshow(plane_to_plot_im.T, vmax=colordeep, vmin=-colordeep, cmap=cmap)
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
			save_dir   = self.directory + foldername

			if os.path.exists(save_dir) == False: os.mkdir(save_dir)
			fig.savefig('%s%s_%s_%s.png' %(save_dir, str(today), what, tstep), format='png')
			plt.close('all')

	"""
	def plot_ref_trs(self, Space, Src,**kwargs):

		self.Space.comm.Barrier()

		if self.Space.MPIrank == self.Space.who_put_src: self.Space.comm.send(self.Space.src, dest=0, tag=1800)
		if self.Space.MPIrank == self.Space.who_get_trs: self.Space.comm.send(self.Space.trs, dest=0, tag=1801)
		if self.Space.MPIrank == self.Space.who_get_ref: self.Space.comm.send(self.Space.ref, dest=0, tag=1802)

		if self.Space.MPIrank == 0 :

			self.Space.src = self.Space.comm.recv( source=self.Space.who_put_src, tag=1800)
			self.Space.trs = self.Space.comm.recv( source=self.Space.who_get_trs, tag=1801)
			self.Space.ref = self.Space.comm.recv( source=self.Space.who_get_ref, tag=1802)

			assert len(self.Space.src.shape) == 1
			assert len(self.Space.ref.shape) == 1
			assert len(self.Space.trs.shape) == 1

			nax = np.newaxis
			time_domain = np.arange(self.Space.tsteps)
			dt = self.Space.dt
			t  = time_domain * dt

			self.Space.src_ft = (dt*self.Space.src[nax,:] \
									* np.exp(1.j*2.*np.pi*Src.freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2.*np.pi)
			self.Space.ref_ft = (dt*self.Space.ref[nax,:] \
									* np.exp(1.j*2.*np.pi*Src.freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2.*np.pi)
			self.Space.trs_ft = (dt*self.Space.trs[nax,:] \
									* np.exp(1.j*2.*np.pi*Src.freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2.*np.pi)

			Trs = (abs(self.Space.trs_ft)**2) / (abs(self.Space.src_ft)**2)
			Ref = (abs(self.Space.ref_ft)**2) / (abs(self.Space.src_ft)**2)

			figsize = (10,7)
			ylim    = 1.1
			Sum     = True

			for key, value in list(kwargs.items()):

				if key == 'figsize': figsize = value
				if key == 'xlim'   : xlim    = value
				if key == 'ylim'   : ylim    = value
				if key == 'Sum'    : Sum= value

			#----------------------------------------------------------------------#
			#------------------------ Plot freq vs ref and trs --------------------#
			#----------------------------------------------------------------------#

			freq_vs_RT = plt.figure(figsize=figsize)
			ax1 = freq_vs_RT.add_subplot(1,1,1)
			ax1.plot(Src.freq.real, Ref.real , color='g', label='Ref')
			ax1.plot(Src.freq.real, Trs.real , color='r', label='Trs')

			if Sum == True :
				total = Trs + Ref
				ax1.plot(Src.freq.real, total.real, color='num_row', label='Trs+Ref')

			ax1.set_xlabel("freq")
			ax1.set_ylabel("Ratio")
			ax1.set_title("%s, Ref,Trs" %(self.Space.where))
			ax1.set_ylim(0, ylim)
			ax1.legend(loc='best')
			ax1.grid(True)

			freq_vs_RT.savefig(self.directory + "freq_vs_Trs_Ref.png")

			#----------------------------------------------------------------------#
			#----------------------- Plot wvlen vs ref and trs --------------------#
			#----------------------------------------------------------------------#

			wvlen_vs_RT = plt.figure(figsize=figsize)
			ax1 = wvlen_vs_RT.add_subplot(1,1,1)
			ax1.plot(Src.wvlen.real, Ref.real , color='g', label='Ref')
			ax1.plot(Src.wvlen.real, Trs.real , color='r', label='Trs')

			if Sum == True :
				total = Trs + Ref
				ax1.plot(Src.wvlen.real, total.real, color='num_row', label='Trs+Ref')

			ax1.set_xlabel("wavelength")
			ax1.set_ylabel("Ratio")
			ax1.set_title("%s, Ref,Trs" %(self.Space.where))
			ax1.set_ylim(0, ylim)
			ax1.legend(loc='best')
			ax1.grid(True)

			wvlen_vs_RT.savefig(self.directory + "wvlen_vs_Trs_Ref.png")

			#----------------------------------------------------------------------#
			#------------------------ Plot time vs ref and trs --------------------#
			#----------------------------------------------------------------------#

		else : pass

		self.Space.comm.Barrier()
		"""

	def plot_src(self, Src, **kwargs):

		"""Plot the input source in time domain and frequency domain.

		PARAMETERs
		----------
		Src: Source object

		kwargs: dictionary
				
				Key				value
				---------		---------
				'figsize'		tuple
				'loc'			string

		RETURNs
		-------
		None
		"""

		if self.Space.MPIrank == self.Space.who_put_src:

				self.Space.comm.send(self.Space.src_re, dest=0, tag=1803)
				self.Space.comm.send(self.Space.src_im, dest=0, tag=1804)

		if self.Space.MPIrank == 0:

			try:
				import matplotlib.pyplot as plt
				from mpl_toolkits.mplot3d import axes3d
				from mpl_toolkits.axes_grid1 import make_axes_locatable

			except ImportError as err:
				print("Please install matplotlib at rank 0")
				sys.exit()

			figsize = (21,10)
			loc     = 'best'

			for key,value in kwargs.items():
				if   key == 'figsize': figsize = value
				elif key == 'loc'	 : loc	   = value

			time_domain = np.arange(self.Space.tsteps,dtype=int)
			nax = np.newaxis
			dt  = self.Space.dt
			t   = time_domain * dt

			src_fig = plt.figure(figsize=figsize)

			ax1 = src_fig.add_subplot(2,3,1)
			ax2 = src_fig.add_subplot(2,3,2)
			ax3 = src_fig.add_subplot(2,3,3)
			ax4 = src_fig.add_subplot(2,3,4)
			ax5 = src_fig.add_subplot(2,3,5)
			ax6 = src_fig.add_subplot(2,3,6)

			label11 = self.Space.where_re + r'$(t)$, real'
			label12 = self.Space.where_im + r'$(t)$, imag'

			label21 = self.Space.where_re + r'$(f)$, real'
			label22 = self.Space.where_re + r'$(f)$, imag'
			label23 = self.Space.where_im + r'$(f)$, real'
			label24 = self.Space.where_im + r'$(f)$, imag'

			label31 = self.Space.where_re + r'$(\lambda)$, real'
			label32 = self.Space.where_re + r'$(\lambda)$, imag'
			label33 = self.Space.where_im + r'$(\lambda)$, real'
			label34 = self.Space.where_im + r'$(\lambda)$, imag'

			label4  = r'$abs(%s(t))$' %(self.Space.where_re[0:2])
			label51 = r'$abs(%s_{re}(f))$' %(self.Space.where_re[0:2])
			label52 = r'$abs(%s_{im}(f))$' %(self.Space.where_im[0:2])
			label61 = r'$abs(%s_{real}(\lambda))$' %(self.Space.where_re[0:2])
			label62 = r'$abs(%s_{imag}(\lambda))$' %(self.Space.where_im[0:2])

			# Source data in time domain.
			self.Space.src_re = self.Space.comm.recv(source=self.Space.who_put_src, tag=1803)
			self.Space.src_im = self.Space.comm.recv(source=self.Space.who_put_src, tag=1804)

			self.Space.src_abs = np.sqrt((self.Space.src_re)**2 + (self.Space.src_im)**2)

			# Source data in frequency domain.
			src_re_dft = (dt*self.Space.src_re[nax,:] *	np.exp(1.j*2.*np.pi*Src.freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2.*np.pi)
			src_im_dft = (dt*self.Space.src_im[nax,:] *	np.exp(1.j*2.*np.pi*Src.freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2.*np.pi)

			self.Space.src_re_dft = src_re_dft
			self.Space.src_im_dft = src_im_dft

			ax1.plot(time_domain    , self.Space.src_re, color='b', label=label11)
			ax1.plot(time_domain    , self.Space.src_im, color='r', label=label12, linewidth='3', alpha=0.3)

			ax2.plot(Src.freq/10**12, self.Space.src_re_dft.real, label=label21)
			ax2.plot(Src.freq/10**12, self.Space.src_re_dft.imag, label=label22)
			ax2.plot(Src.freq/10**12, self.Space.src_im_dft.real, label=label23, linewidth='5', alpha=0.3)
			ax2.plot(Src.freq/10**12, self.Space.src_im_dft.imag, label=label24, linewidth='5', alpha=0.3)

			ax3.plot(Src.wvlen*10**9, self.Space.src_re_dft.real, label=label31)
			ax3.plot(Src.wvlen*10**9, self.Space.src_re_dft.imag, label=label32)
			ax3.plot(Src.wvlen*10**9, self.Space.src_im_dft.real, label=label33, linewidth='5', alpha=0.3)
			ax3.plot(Src.wvlen*10**9, self.Space.src_im_dft.imag, label=label34, linewidth='5', alpha=0.3)

			ax4.plot(time_domain    , self.Space.src_abs, color='b', label=label4)
			ax5.plot(Src.freq/10**12, abs(src_re_dft)**2, label=label51)
			ax5.plot(Src.freq/10**12, abs(src_im_dft)**2, linewidth='4', alpha=0.3, label=label52)
			ax6.plot(Src.wvlen*10**9, abs(src_re_dft)**2, label=label61)
			ax6.plot(Src.wvlen*10**9, abs(src_im_dft)**2, linewidth='4', alpha=0.3, label=label62)

			ax1.set_xlabel("time step")
			ax1.set_ylabel("Amp")
			ax1.legend(loc=loc)
			ax1.grid(True)

			ax2.set_xlabel("freq(THz)")
			ax2.set_ylabel("Amp")
			ax2.legend(loc=loc)
			ax2.grid(True)

			ax3.set_xlabel("wvlen(nm)")
			ax3.set_ylabel("Amp")
			ax3.legend(loc=loc)
			ax3.grid(True)

			ax4.set_xlabel("time step")
			ax4.set_ylabel("Intensity")
			ax4.legend(loc=loc)
			ax4.grid(True)

			ax5.set_xlabel("freq(THz)")
			ax5.set_ylabel("Intensity")
			ax5.legend(loc=loc)
			ax5.grid(True)

			ax6.set_xlabel("wvlen(nm)")
			ax6.set_ylabel("Intensity")
			ax6.legend(loc=loc)
			ax6.grid(True)

			src_fig.savefig(self.directory+"simulated_source.png")
