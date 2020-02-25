# Author  : Ki-Hwan Kim
# Purpose : Update function for a Drude-type metal (single pole)
# Target  : GPU using PyOpenCL
# Created : 2012-01-30
# Modified: 

import numpy as np
import pyopencl as cl

from kemp.fdtd3d.util import common, common_gpu
from fields import Fields


class Drude:
    def __init__(self, fields, pt0, pt1, ep_inf, drude_freq, gamma, mask_arrays=(1,1,1)):
        common.check_type('fields', fields, Fields)
        common.check_type('pt0', pt0, (list, tuple), (int, float))
        common.check_type('pt1', pt1, (list, tuple), (int, float))
        common.check_type('ep_inf', ep_inf, (int, float))
        common.check_type('drude_freq', drude_freq, (int, float))
        common.check_type('gamma', gamma, (int, float))
        common.check_type('mask_arrays', mask_arrays, (list, tuple), (np.ndarray, int))

        # local variables
        pt0 = common.convert_indices(fields.ns, pt0)
        pt1 = common.convert_indices(fields.ns, pt1)
        context = fields.context
        queue = fields.queue
        dtype = fields.dtype
        shape = common.shape_two_points(pt0, pt1, is_dummy=True)
        
        for axis, n, p0, p1 in zip(['x', 'y', 'z'], fields.ns, pt0, pt1):
            common.check_value('pt0 %s' % axis, p0, range(n))
            common.check_value('pt1 %s' % axis, p1, range(n))
        
        for mask_array in mask_arrays:
            if isinstance(mask_array, np.ndarray):
                assert common.shape_two_points(pt0, pt1) == mask_array.shape, \
                       'shape mismatch : %s, %s' % (shape, mask_array.shape)
            
        # allocations
        psis = [np.zeros(shape, dtype) for i in range(3)]
        psi_bufs = [cl.Buffer(context, cl.mem_flags.READ_WRITE, psi.nbytes) for psi in psis]
        for psi_buf, psi in zip(psi_bufs, psis): cl.enqueue_copy(queue, psi_buf, psi)
        
        dt = fields.dt
        aa = (2 - gamma * dt) / (2 + gamma * dt)
        bb = drude_freq**2 * dt / (2 + gamma * dt)
        comm = 2 * ep_inf + bb * dt
        ca = 2 * dt / comm
        cb = - (aa + 3) * bb * dt / comm
        cc = - (aa + 1) * dt / comm
        cas = [ca * mask for mask in mask_arrays]
        
        shape = common.shape_two_points(pt0, pt1, is_dummy=True)
        f = np.zeros(shape, dtype)
        psi_bufs = [cl.Buffer(context, cl.mem_flags.READ_WRITE, f.nbytes) for i in range(3)]
        for psi_buf in psi_bufs: cl.enqueue_copy(queue, psi_buf, f)
        
        cf = np.ones(shape, dtype)
        mask_bufs = [cl.Buffer(context, cl.mem_flags.READ_ONLY, cf.nbytes) for i in range(3)]
        for mask_buf, mask in zip(mask_bufs, mask_arrays): cl.enqueue_copy(queue, mask_buf, cf * mask)
        
        # modify ce arrays
        slices = common.slices_two_points(pt0, pt1)
        for ce, ca in zip(fields.get_ces(), cas):
            ce[slices] = ca * mask + ce[slices] * mask.__invert__()
        
        # program
        nmax_str, xid_str, yid_str, zid_str = common_gpu.macro_replace_list(pt0, pt1)
        macros = ['NMAX', 'XID', 'YID', 'ZID', 'DX', 'DTYPE', 'PRAGMA_fp64']
        values = [nmax_str, xid_str, yid_str, zid_str, str(fields.ls)] + fields.dtype_str_list
        
        ksrc = common.replace_template_code( \
            open(common_gpu.src_path + 'drude.cl').read(), macros, values)
        program = cl.Program(fields.context, ksrc).build()
            
        # arguments
        pca = aa
        pcb = (aa + 1) * bb
        args = fields.ns + [dtype(cb), dtype(cc), dtype(pca), dtype(pcb)] \
            + fields.eh_bufs[:3] + psi_bufs + mask_bufs
        
        # global variables
        self.mainf = fields
        self.program = program
        self.args = args

        nx, ny, nz = fields.ns
        nmax = int(nmax_str)
        remainder = nmax % fields.ls
        self.gs = nmax if remainder == 0 else nmax - remainder + fields.ls 
        
        # append to the update list
        self.priority_type = 'material'
        fields.append_instance(self)


    def update_e(self):
        self.program.update_e(self.mainf.queue, (self.gs,), (self.mainf.ls,), *self.args)


    def update_h(self):
        pass
