import numpy as np

from kemp.fdtd3d.util import common
from fields import Fields


class Pbc:
    def __init__(self, fields, axes):
        common.check_type('fields', fields, Fields)
        common.check_type('axes', axes, str)

        assert len( set(axes).intersection(set('xyz')) ) > 0

        # global variables
        self.mainf = fields
        self.axes = axes

        # append to the update list
        self.priority_type = 'pbc'
        fields.append_instance(self)


    def update_e(self):
        ex, ey, ez, hx, hy, hz = self.mainf.ehs
        axes = self.axes

        if 'x' in axes:
            ey[-1,:,:] = ey[0,:,:]
            ez[-1,:,:] = ez[0,:,:] 

        if 'y' in axes:
            ex[:,-1,:] = ex[:,0,:]
            ez[:,-1,:] = ez[:,0,:] 

        if 'z' in axes:
            ex[:,:,-1] = ex[:,:,0]
            ey[:,:,-1] = ey[:,:,0]


    def update_h(self):
        ex, ey, ez, hx, hy, hz = self.mainf.ehs
        axes = self.axes

        if 'x' in axes:
            hy[0,:,:] = hy[-1,:,:]
            hz[0,:,:] = hz[-1,:,:]

        if 'y' in axes:
            hx[:,0,:] = hx[:,-1,:]
            hz[:,0,:] = hz[:,-1,:]

        if 'z' in axes:
            hx[:,:,0] = hx[:,:,-1]
            hy[:,:,0] = hy[:,:,-1]
