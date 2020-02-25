import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os, datetime
from build import Fields

class Graphtool(object):

	def __init__(self,path):

		savedir = path + 'graph/'

		if not os.path.exists(path)    : os.mkdir(path)
		if not os.path.exists(savedir) : os.mkdir(savedir)

		self._directory = savedir 

		return None
		
	def plot2D3D(self, Fields, what, step, xidx=None, yidx=None, zidx=None,**kwargs):
		"""Plot 2D and 3D graph for given field and position

		Parameters
		------------
		field : ndarray
			numpy array given by Fields object
		"""

		self.figsize = (21,9)
		self.cmap = plt.cm.bwr
		stride    = 1
		color     = 'b'

		if what == 'Ex' or what == 'Ey' or what == 'Ez':
			zlim      = 2. 
			colordeep = 2.

		if what == 'Hx' or what == 'Hy' or what == 'Hz':
			zlim      = 0.01 
			colordeep = 0.01

		for key, value in list(kwargs.items()):
			if key == 'colordeep' : colordeep    = value
			elif key == 'stride'  : stride       = value
			elif key == 'zlim'    : zlim         = value
			elif key == 'figsize' : self.figsize = value
			elif key == 'cmap'    : self.cmap    = value
			elif key == 'color'   : color        = value

		Fields.comm.Barrier()
		
		###################################################################################
		###################### Gather field data from all slave nodes #####################
		###################################################################################
		
		if   what == 'Ex' : self.gathered_fields = Fields.comm.gather(Fields.Ex, root=0)
		elif what == 'Ey' : self.gathered_fields = Fields.comm.gather(Fields.Ey, root=0)
		elif what == 'Ez' : self.gathered_fields = Fields.comm.gather(Fields.Ez, root=0)
		elif what == 'Hx' : self.gathered_fields = Fields.comm.gather(Fields.Hx, root=0)
		elif what == 'Hy' : self.gathered_fields = Fields.comm.gather(Fields.Hy, root=0)
		elif what == 'Hz' : self.gathered_fields = Fields.comm.gather(Fields.Hz, root=0)

		if Fields.rank == 0 : 

			assert len(self.gathered_fields) == Fields.size

			if xidx != None : 
				assert type(xidx) == int
				yidx  = slice(None,None) # indices from beginning to end
				zidx  = slice(None,None)
				plane = 'yz'
				a = np.arange(Fields.gridy)
				b = np.arange(Fields.gridz)
				plotfield = np.zeros((len(a),len(b)), dtype=Fields.dtype)

			elif yidx != None :
				assert type(yidx) == int
				xidx  = slice(None,None)
				zidx  = slice(None,None)
				plane = 'xz'
				a = np.arange(Fields.gridx)
				b = np.arange(Fields.gridz)
				plotfield = np.zeros((len(a),len(b)), dtype=Fields.dtype)

			elif zidx != None :
				assert type(zidx) == int
				xidx  = slice(None,None)
				yidx  = slice(None,None)
				plane = 'xy'
				a = np.arange(Fields.gridx)
				b = np.arange(Fields.gridy)
				plotfield = np.zeros((len(a),len(b)), dtype=Fields.dtype)
		
			elif (xidx,yidx,zidx) == (None,None,None):
				raise ValueError("Plane is not defined. Please insert one of x,y or z index of the plane.")

			#####################################################################################
			######### Build up total field with the parts of the grid from slave nodes ##########
			#####################################################################################

			integrated_field = np.zeros((Fields.grid), dtype=Fields.dtype)
			
			if what == 'Ex' or what == 'Ey' or what == 'Hz':

				for part in range(1,len(self.gathered_fields)):

					plot_index = slice(0,-1)
					if part == (len(self.gathered_fields)-1) : plot_index = slice(0,None) 
					portion = Fields.OWNslices_of_slaves[part]
					integrated_field[:,:,portion] = self.gathered_fields[part][:,:,plot_index]

			elif what == 'Ez' or what == 'Hx' or what == 'Hy':

				for part in range(1,len(self.gathered_fields)):

					plot_index = slice(1,None)
					if part == 1: plot_index = slice(0,None) 
					portion = Fields.OWNslices_of_slaves[part]
					integrated_field[:,:,portion] = self.gathered_fields[part][:,:,plot_index]

			plotfield = integrated_field[xidx, yidx, zidx]

			B,A = np.meshgrid(a,b)
			today = datetime.date.today()

			fig = plt.figure(figsize=self.figsize)
			ax1 = fig.add_subplot(1,2,1)
			ax2 = fig.add_subplot(1,2,2, projection='3d')

			ax1.set_title('%s 2D' %what)
			ax2.set_title('%s 3D' %what)
			ax2.set_zlim(-zlim,zlim)
			ax2.set_zlabel('field')

			if plane == 'yz':

				A = A[::-1]
				im = ax1.imshow(plotfield[:,:].real, vmax=colordeep, vmin=-colordeep, cmap=self.cmap)
				ax2.plot_wireframe(A,B,plotfield[B,A].real, color=color, rstride=stride, cstride=stride)
				divider = make_axes_locatable(ax1)
				cax = divider.append_axes('right', size='5%', pad=0.1)
				cbar = fig.colorbar(im, cax=cax)

				ax1.set_xlabel('z')
				ax1.set_ylabel('y')
				ax2.set_xlabel('z')
				ax2.set_ylabel('y')

			elif plane == 'xy':

				A = A[::-1]
				im = ax1.imshow(plotfield[B,A].real, vmax=colordeep, vmin=-colordeep, cmap=self.cmap)
				ax2.plot_wireframe(B,A,plotfield[B,A].real, color=color, rstride=stride, cstride=stride)
				divider = make_axes_locatable(ax1)
				cax = divider.append_axes('right', size='5%', pad=0.1)
				cbar = fig.colorbar(im, cax=cax)

				ax1.set_xlabel('x')
				ax1.set_ylabel('y')
				ax2.set_xlabel('x')
				ax2.set_ylabel('y')

			elif plane == 'xz':

				A = A[::-1]
				im = ax1.imshow(plotfield[:,:].real, vmax=colordeep, vmin=-colordeep, cmap=self.cmap)
				ax2.plot_wireframe(A,B,plotfield[B,A].real, color=color, rstride=stride, cstride=stride)
				divider = make_axes_locatable(ax1)
				cax = divider.append_axes('right', size='5%', pad=0.1)
				cbar = fig.colorbar(im, cax=cax)

				ax1.set_xlabel('z')
				ax1.set_ylabel('x')
				ax2.set_xlabel('z')
				ax2.set_ylabel('x')

			foldername = 'plot2D3D/'
			save_dir = self._directory + foldername
			if not os.path.exists(save_dir) : os.mkdir(save_dir)
			fig.savefig('%s%s_%s_%s.png' %(save_dir, str(today), what, step))
			plt.close('all')

		else : pass

		return None

#	def plot3D(self):
#
#	def plot2D(self):
#
#	def plotSrc(self):
#
	def plot_rRef(self,Fields,**kwargs):

		Fields.comm.Barrier()

		if   Fields.rank == Fields.who_get_ref: Fields.comm.send(Fields.ref, dest=0, tag=17)
		elif Fields.rank == 0 :
			
			Fields.ref = Fields.comm.recv( source=Fields.who_put_src, tag=17)

			figsize = (10,7)
			color   = 'blue'
			xlim	= Fields.nsteps
			ylim    = 2.1

			for key, value in list(kwargs.items()):

				if key == 'figsize' : figsize = value
				if key == 'color'   : color   = value
				if key == 'xlim'    : xlim    = value
				if key == 'ylim'    : ylim    = value

			fig = plt.figure(figsize=figsize)
			
			time_domain = np.arange(Fields.nsteps)
		
			ax1 = fig.add_subplot(1,1,1)
			ax1.plot(time_domain,Fields.ref.real , color=color, label='Ref')
			ax1.set_xlabel("time steps")
			ax1.set_ylabel("Amp")
			ax1.set_title("%s, Ref" %(Fields.where))
			ax1.set_xlim(0,xlim)
			ax1.set_ylim(-ylim, ylim)

			ax1.legend(loc='best')
			ax1.grid(True)

			fig.savefig(self._directory + "Ref.png")

		else : pass

		Fields.comm.Barrier()

		return None

	def plot_trs(self,Fields,**kwargs):
	
		Fields.comm.Barrier()

		if Fields.rank   == Fields.who_get_trs	: Fields.comm.send(Fields.trs, dest=0, tag=18)
		elif Fields.rank == 0 :

			Fields.trs = Fields.comm.recv( source=Fields.who_get_trs, tag=18)

			figsize = (10,7)
			color   = 'blue'
			xlim	= Fields.nsteps
			ylim    = 2.1

			for key, value in list(kwargs.items()):

				if key == 'figsize' : figsize = value
				if key == 'color'   : color   = value
				if key == 'xlim'    : xlim    = value
				if key == 'ylim'    : ylim    = value

			fig = plt.figure(figsize=figsize)

			time_domain = np.arange(Fields.nsteps)

			ax1 = fig.add_subplot(1,1,1)
			ax1.plot(time_domain, Fields.trs.real, color=color, label='Trs')
			ax1.set_xlabel("time steps")
			ax1.set_ylabel("Amp")
			ax1.set_title("%s, Trs" %(Fields.where))
			ax1.set_xlim(0,xlim)
			ax1.set_ylim(-ylim, ylim)

			ax1.legend(loc='best')
			ax1.grid(True)

			fig.savefig(self._directory + "Trs.png")

		else : pass

		Fields.comm.Barrier()

		return None

	def plot_ref_trs(self,Fields,Src,**kwargs):

		Fields.comm.Barrier()

		if Fields.rank == Fields.who_put_src: Fields.comm.send(Fields.src, dest=0, tag=1800)
		if Fields.rank == Fields.who_get_trs: Fields.comm.send(Fields.trs, dest=0, tag=1801)
		if Fields.rank == Fields.who_get_ref: Fields.comm.send(Fields.ref, dest=0, tag=1802)

		if Fields.rank == 0 :

			Fields.src = Fields.comm.recv( source=Fields.who_put_src, tag=1800)
			Fields.trs = Fields.comm.recv( source=Fields.who_get_trs, tag=1801)
			Fields.ref = Fields.comm.recv( source=Fields.who_get_ref, tag=1802)

			assert len(Fields.src.shape) == 1
			assert len(Fields.ref.shape) == 1
			assert len(Fields.trs.shape) == 1

			nax = np.newaxis
			tsteps = np.arange(Fields.nsteps)
			dt = Fields.dt
			t  = tsteps * dt

			Fields.src_ft = (dt*Fields.src[nax,:] * np.exp(1.j*2.*np.pi*Src.freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2.*np.pi)
			Fields.ref_ft = (dt*Fields.ref[nax,:] * np.exp(1.j*2.*np.pi*Src.freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2.*np.pi)
			Fields.trs_ft = (dt*Fields.trs[nax,:] * np.exp(1.j*2.*np.pi*Src.freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2.*np.pi)

			Trs = (abs(Fields.trs_ft)**2) / (abs(Fields.src_ft)**2)
			Ref = (abs(Fields.ref_ft)**2) / (abs(Fields.src_ft)**2)

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
				ax1.plot(Src.freq.real, total.real, color='b', label='Trs+Ref')

			ax1.set_xlabel("freq")
			ax1.set_ylabel("Ratio")
			ax1.set_title("%s, Ref,Trs" %(Fields.where))
			ax1.set_ylim(0, ylim)
			ax1.legend(loc='best')
			ax1.grid(True)

			freq_vs_RT.savefig(self._directory + "freq_vs_Trs_Ref.png")

			#----------------------------------------------------------------------#
			#----------------------- Plot wvlen vs ref and trs --------------------#
			#----------------------------------------------------------------------#

			wvlen_vs_RT = plt.figure(figsize=figsize)
			ax1 = wvlen_vs_RT.add_subplot(1,1,1)
			ax1.plot(Src.wvlen.real, Ref.real , color='g', label='Ref')
			ax1.plot(Src.wvlen.real, Trs.real , color='r', label='Trs')

			if Sum == True :
				total = Trs + Ref
				ax1.plot(Src.wvlen.real, total.real, color='b', label='Trs+Ref')

			ax1.set_xlabel("wavelength")
			ax1.set_ylabel("Ratio")
			ax1.set_title("%s, Ref,Trs" %(Fields.where))
			ax1.set_ylim(0, ylim)
			ax1.legend(loc='best')
			ax1.grid(True)

			wvlen_vs_RT.savefig(self._directory + "wvlen_vs_Trs_Ref.png")

			#----------------------------------------------------------------------#
			#------------------------ Plot time vs ref and trs --------------------#
			#----------------------------------------------------------------------#

		else : pass

		Fields.comm.Barrier()

		return None

	def plot_src(self, Fields,Src,**kwargs):
		
		Fields.comm.Barrier()

		if Fields.rank == Fields.who_put_src: Fields.comm.send(Fields.src, dest=0, tag=1803)

		if Fields.rank == 0:

			figsize = (15,7)
			loc     = 'best'

			for key,value in kwargs.items():
				if   key == 'figsize': figsize = value
				elif key == 'loc'	 : loc	   = value

			nax = np.newaxis
			tsteps = np.arange(Fields.nsteps,dtype=int)
			dt = Fields.dt
			t  = tsteps * dt

			Fields.src    = Fields.comm.recv( source=Fields.who_put_src, tag=1803)
			Fields.src_ft = (dt*Fields.src[nax,:] * np.exp(1.j*2.*np.pi*Src.freq[:,nax]*t[nax,:])).sum(1) / np.sqrt(2.*np.pi)

			src_fig = plt.figure(figsize=figsize)

			ax1 = src_fig.add_subplot(1,2,1)
			ax2 = src_fig.add_subplot(1,2,2)

			label1 = Fields.where + r'$(t)$, real'
			label2 = Fields.where + r'$(t)$, imag'
			label3 = Fields.where + r'$(f)$'

			src_freq_domain = (abs(Fields.src_ft)**2).real

			ax1.plot(tsteps, Fields.src.real, color='b', label=label1)
			ax1.plot(tsteps, Fields.src.imag, color='r', label=label2)
			ax2.plot(Src.freq.real, src_freq_domain, color='b', label=label3)

			ax1.set_xlabel("time step")
			ax2.set_xlabel("freq")

			ax1.set_ylabel(label1)
			ax2.set_ylabel(label2)

			ax1.legend(loc=loc)
			ax2.legend(loc=loc)

			ax1.grid(True)
			ax2.grid(True)

			src_fig.savefig(self._directory+"src_fig.png")

			return
