# Author  : Ki-Hwan Kim
# Purpose : Update function for a Drude-type metal (single pole)
# Target  : CPU using Numpy
# Created : 2012-01-27
# Modified: 

import numpy as np
import types

from kemp.fdtd3d.util import common
from fields import Fields


class Drude:
    def __init__(self, fields, pt0, pt1, ep_inf, drude_freq, gamma, mask_arrays=(1,1,1)):
        common.check_type('fields', fields, Fields)
        common.check_type('pt0', pt0, (list, tuple), (int, float))
        common.check_type('pt1', pt1, (list, tuple), (int, float))
        common.check_type('ep_inf', ep_inf, (int, float))
        common.check_type('drude_freq', drude_freq, (int, float))
        common.check_type('gamma', gamma, (int, float))
        common.check_type('mask_arrays', mask_arrays, (list, tuple), (np.ndarray, types.IntType))

        # local variables
        pt0 = common.convert_indices(fields.ns, pt0)
        pt1 = common.convert_indices(fields.ns, pt1)
        dtype = fields.dtype
        
        for axis, n, p0, p1 in zip(['x', 'y', 'z'], fields.ns, pt0, pt1):
            common.check_value('pt0 %s' % axis, p0, range(n))
            common.check_value('pt1 %s' % axis, p1, range(n))
        
        for mask_array in mask_arrays:
            if isinstance(mask_array, np.ndarray):
                assert common.shape_two_points(pt0, pt1) == mask_array.shape, \
                       'shape mismatch : %s, %s' % (shape, mask_array.shape)
            
        # allocations
        shape = common.shape_two_points(pt0, pt1, is_dummy=True)
        psis = [np.zeros(shape, dtype) for i in range(3)]
        
        dt = fields.dt
        aa = (2 - gamma * dt) / (2 + gamma * dt)
        bb = drude_freq**2 * dt / (2 + gamma * dt)
        comm = 2 * ep_inf + bb * dt
        ca = 2 * dt / comm
        cb = - (aa + 3) * bb * dt / comm
        cc = - (aa + 1) * dt / comm
        cas = [ca * mask for mask in mask_arrays]
        cbs = [cb * mask for mask in mask_arrays]
        ccs = [cc * mask for mask in mask_arrays]
        
        # modify ce arrays
        slices = common.slices_two_points(pt0, pt1)
        for ce, ca in zip(fields.get_ces(), cas):
            ce[slices] = ca
        
        # global variables
        self.mainf = fields
        self.psis = psis
        self.cbs = cbs
        self.ccs = ccs
        self.pcs = aa, (aa + 1) * bb
        self.slices = slices

        # append to the update list
        self.priority_type = 'material'
        fields.append_instance(self)


    def update_e(self):
        pca, pcb = self.pcs
        sl = self.slices
        
        for f, psi, cb, cc in zip(self.mainf.ehs[:3], self.psis, self.cbs, self.ccs):
            f[sl] += cb * f[sl] + cc * psi[:]
            psi[:] = pca * psi[:] + pcb * f[sl]


    def update_h(self):
        pass