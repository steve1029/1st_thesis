import numpy as np
import os, datetime, sys
from scipy.constants import c

class Graphtool(object):

    def __init__(self, Space, name, path):

        self.Space = Space
        self.name = name
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

    def gather(self, what):
        """
        Gather the data resident in rank >0 to rank 0.
        """
        ###################################################################################
        ###################### Gather field data from all slave nodes #####################
        ###################################################################################
        
        if   what == 'Ex': gathered = self.Space.MPIcomm.gather(self.Space.Ex_re, root=0)
        elif what == 'Ey': gathered = self.Space.MPIcomm.gather(self.Space.Ey_re, root=0)
        elif what == 'Ez': gathered = self.Space.MPIcomm.gather(self.Space.Ez_re, root=0)
        elif what == 'Hx': gathered = self.Space.MPIcomm.gather(self.Space.Hx_re, root=0)
        elif what == 'Hy': gathered = self.Space.MPIcomm.gather(self.Space.Hy_re, root=0)
        elif what == 'Hz': gathered = self.Space.MPIcomm.gather(self.Space.Hz_re, root=0)

        self.what = what

        if self.Space.MPIrank == 0: 
        
            self.integrated = np.zeros((self.Space.grid), dtype=self.Space.dtype)

            for MPIrank in range(self.Space.MPIsize):
                self.integrated[self.Space.myNx_slices[MPIrank],:,:] = gathered[MPIrank]

                #if MPIrank == 1: print(MPIrank, gathered[MPIrank][xidx,yidx,zidx])

            return self.integrated

        else: return None

    def plot2D3D(self, integrated, tstep, xidx=None, yidx=None, zidx=None, **kwargs):

        if self.Space.MPIrank == 0: 

            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import axes3d
                from mpl_toolkits.axes_grid1 import make_axes_locatable

            except ImportError as err:
                print("Please install matplotlib at rank 0")
                sys.exit()

            colordeep = .1
            stride    = 1
            zlim      = 1
            figsize   = (18, 8)
            cmap      = plt.cm.bwr
            lc = 'b'
            aspect = 'auto'

            for key, value in kwargs.items():

                if   key == 'colordeep': colordeep = value
                elif key == 'figsize': figsize = value
                elif key == 'aspect': aspect = value
                elif key == 'stride': stride = value
                elif key == 'what': self.what = value
                elif key == 'zlim': zlim = value
                elif key == 'cmap': cmap = value
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

            self.plane_to_plot_re = integrated[xidx, yidx, zidx].copy()

            Row, Col = np.meshgrid(row, col, indexing='xy', sparse=True)
            today    = datetime.date.today()

            fig  = plt.figure(figsize=figsize)
            ax11 = fig.add_subplot(1,2,1)
            ax12 = fig.add_subplot(1,2,2, projection='3d')

            if plane == 'yz':

                image11 = ax11.imshow(self.plane_to_plot_re.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
                ax12.plot_wireframe(Col, Row, self.plane_to_plot_re[Col, Row], color=lc, rstride=stride, cstride=stride)

                divider11 = make_axes_locatable(ax11)

                cax11  = divider11.append_axes('right', size='5%', pad=0.1)
                cbar11 = fig.colorbar(image11, cax=cax11)

                ax11.invert_yaxis()
                #ax12.invert_yaxis()

                ax11.set_xlabel('y')
                ax11.set_ylabel('z')
                ax12.set_xlabel('y')
                ax12.set_ylabel('z')

            elif plane == 'xy':

                image11 = ax11.imshow(self.plane_to_plot_re.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
                ax12.plot_wireframe(Col, Row, self.plane_to_plot_re[Col, Row], color=lc, rstride=stride, cstride=stride)

                divider11 = make_axes_locatable(ax11)

                cax11  = divider11.append_axes('right', size='5%', pad=0.1)
                cbar11 = fig.colorbar(image11, cax=cax11)

                ax11.invert_yaxis()
                #ax12.invert_yaxis()

                ax11.set_xlabel('x')
                ax11.set_ylabel('y')
                ax12.set_xlabel('x')
                ax12.set_ylabel('y')

            elif plane == 'xz':

                image11 = ax11.imshow(self.plane_to_plot_re.T, vmax=colordeep, vmin=-colordeep, cmap=cmap, aspect=aspect)
                ax12.plot_wireframe(Col, Row, self.plane_to_plot_re[Col, Row], color=lc, rstride=stride, cstride=stride)

                divider11 = make_axes_locatable(ax11)

                cax11  = divider11.append_axes('right', size='5%', pad=0.1)
                cbar11 = fig.colorbar(image11, cax=cax11)

                #ax11.invert_yaxis()
                ax12.invert_yaxis()

                ax11.set_xlabel('x')
                ax11.set_ylabel('z')
                ax12.set_xlabel('x')
                ax12.set_ylabel('z')

            ax11.set_title(r'$%s.real, 2D$' %self.what)
            ax12.set_title(r'$%s.real, 3D$' %self.what)

            ax12.set_zlim(-zlim,zlim)
            ax12.set_zlabel('field')

            foldername = 'plot2D3D/'
            save_dir   = self.savedir + foldername

            if os.path.exists(save_dir) == False: os.mkdir(save_dir)
            plt.tight_layout()
            fig.savefig('%s%s_%s_%s_%s_%s.png' %(save_dir, str(today), self.name,self.what, plane, tstep), format='png', bbox_inches='tight')
            plt.close('all')
