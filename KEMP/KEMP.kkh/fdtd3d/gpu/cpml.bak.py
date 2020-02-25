# Author  : Ki-Hwan Kim
# Purpose : Update function for the CPML (Convolutional Perfectly Matched Layer)
# Target  : GPU using PyOpenCL
# Created : 2012-01-30
# Modified: 

import numpy as np
import pyopencl as cl

from kemp.fdtd3d.util import common, common_gpu
from fields import Fields


class Pml:
    def __init__(self, fields, directions, npml=10, sigma_max=4, kappa_max=1, alpha_max=0, m_sigma=3, m_alpha=1):
        """
        """

        common.check_type('fields', fields, Fields)
        common.check_type('directions', directions, (list, tuple), str)
        common.check_type('npml', npml, int)
        common.check_type('sigma_max', sigma_max, (int, float))
        common.check_type('kappa_max', kappa_max, (int, float))
        common.check_type('alpha_max', alpha_max, (int, float))
        common.check_type('m_sigma', m_sigma, (int, float))
        common.check_type('m_alpha', m_alpha, (int, float))

        assert len(directions) == 3
        for axis in directions:
            assert axis in ['+', '-', '+-', '']

        # local variables
        dt = fields.dt
        nx, ny, nz = fields.ns
        dtype = fields.dtype
        context = fields.context
        queue = fields.queue

        # allocations
        fx = np.zeros((2*npml, ny, nz), dtype)
        fy = np.zeros((nx, 2*npml, nz), dtype)
        fz = np.zeros((nx, ny, 2*npml), dtype)
        psix_bufs = [cl.Buffer(context, cl.mem_flags.READ_WRITE, fx.nbytes) for i in range(4)]
        psiy_bufs = [cl.Buffer(context, cl.mem_flags.READ_WRITE, fy.nbytes) for i in range(4)]
        psiz_bufs = [cl.Buffer(context, cl.mem_flags.READ_WRITE, fz.nbytes) for i in range(4)]
        for psi_bufs, f in zip([psix_bufs, psiy_bufs, psiz_bufs], [fx, fy, fz]):
            for psi_buf in psi_bufs:
                cl.enqueue_copy(queue, psi_buf, f)
        del fx, fy, fz

        i_half = np.arange(0.5, npml, 1, dtype)
        i_one = np.arange(1, npml+1, 1, dtype)
        sigma_half = sigma_max * (i_half / npml) ** m_sigma
        sigma_one = sigma_max * (i_one / npml) ** m_sigma
        kappa_half = 1 + (kappa_max - 1) * (i_half / npml) ** m_sigma
        kappa_one = 1 + (kappa_max - 1) * (i_one / npml) ** m_sigma
        alpha_half = alpha_max * ((npml - i_half) / npml) ** m_alpha
        alpha_one = alpha_max * ((npml - i_one) / npml) ** m_alpha
        pcb_half = np.exp(-(sigma_half / kappa_half + alpha_half) * dt)
        pcb_one = np.exp(-(sigma_one / kappa_one + alpha_one) * dt)
        pca_half = sigma_half / (sigma_half + alpha_half * kappa_half) * (pcb_half - 1)
        pca_one = sigma_one / (sigma_one + alpha_one * kappa_one) * (pcb_one - 1)

        pc_bufs = [cl.Buffer(context, cl.mem_flags.READ_ONLY, pcb_half.nbytes) for i in range(4)]
        for pc_buf, pc in zip(pc_bufs, [pcb_half, pcb_one, pca_half, pca_one]): 
            cl.enqueue_copy(queue, pc_buf, pc)

        # modify reciprocal ds
        if kappa_max != 1:
            erds, hrds = fields.get_rds()
            for erd, hrd, pms in zip(erds, hrds, directions):
                if '+' in pms:
                    erd[-npml-1:-1] /= kappa_half
                    hrd[-npml:] /= kappa_one
                if '-' in pms:
                    erd[:npml] /= kappa_one
                    hrd[1:npml+1] /= kappa_half
        
        # program
        nmax = {'x': 'npml*ny*nz', 'y': 'nx*npml*nz', 'z': 'nx*ny*npml'}
        idx_psi = {'+': 'idx+nmax', '-': 'idx'}
        idx_pc = {'x': {'+': 'idx/(ny*nz)', '-': 'npml-1-idx/(ny*nz)'}, \
                  'y': {'+': '(idx/nz)%npml', '-': 'npml-1-(idx/nz)%npml'}, \
                  'z': {'+': 'idx%npml', '-': 'npml-1-idx%npml'}}
        if0 = {'x': 'idx', 'y': 'idx+(idx/(npml*nz))*(ny-npml)*nz', 'z': 'idx+(idx/npml)*(nz-npml)'}
        idx_f1 = {'E': {'x': {'+': if0['x']+'+(nx-npml-1)*ny*nz', '-': if0['x']}, \
                        'y': {'+': if0['y']+'+(ny-npml-1)*nz', '-': if0['y']}, \
                        'z': {'+': if0['z']+'+(nz-npml-1)', '-': if0['z']}}, \
                  'H': {'x': {'+': if0['x']+'+(nx-npml)*ny*nz', '-': if0['x']+'+ny*nz'}, \
                        'y': {'+': if0['y']+'+(ny-npml)*nz', '-': if0['y']+'+nz'}, \
                        'z': {'+': if0['z']+'+(nz-npml)', '-': if0['z']+'+1'}}}
        idx_f2 = {'E': {'x': 'if1+ny*nz', 'y': 'if1+nz', 'z': 'if1+1'}, \
                  'H': {'x': 'if1', 'y': 'if1', 'z': 'if1'}}
        idx_f3 = {'E': {'x': 'if1', 'y': 'if1', 'z': 'if1'}, \
                  'H': {'x': 'if1-ny*nz', 'y': 'if1-nz', 'z': 'if1-1'}}

        values_const = ['', '0.5', '0.5']
        values_array = [', __global DTYPE *cf1, __global DTYPE *cf2', 'cf1[if1]', 'cf2[if1]']

        programs = {}
        for xyz in ['x', 'y', 'z']:
            programs[xyz] = {}
            for eh in ['E', 'H']:
                programs[xyz][eh] = {}
                for pm in ['+', '-']:
                    macros = ['NMAX', 'IDX_PSI', 'IDX_PC', 'IDX_F1', 'IDX_F2', 'IDX_F3', 'ARGS_CF', 'CF1', 'CF2', 'DTYPE', 'PRAGMA_fp64']
                    values = [nmax[xyz], idx_psi[pm], idx_pc[xyz][pm], idx_f1[eh][xyz][pm], idx_f2[eh][xyz], idx_f3[eh][xyz]]
                    if (eh == 'E' and fields.ce_on) or (eh == 'H' and fields.ch_on):
                        values += values_array
                    else:
                        values += values_const
                    values += fields.dtype_str_list
                    ksrc = common.replace_template_code(open(common_gpu.src_path + 'cpml.cl').read(), macros, values)
                    programs[xyz][eh][pm] = cl.Program(context, ksrc).build()

        # arguments
        ex, ey, ez, hx, hy, hz = fields.eh_bufs
        if fields.ce_on:
            cex, cey, cez = fields.ce_bufs
            ces = {'x': [cey, cez], 'y': [cez, cex], 'z': [cex, cey]}
        else:
            ces = {'x': [], 'y': [], 'z': []}
        if fields.ch_on:
            chx, chy, chz = fields.ch_bufs
            chs = {'x': [chz, chy], 'y': [chx, chz], 'z': [chy, chx]}
        else:
            chs = {'x': [], 'y': [], 'z': []}
        psi_eyx, psi_ezx, psi_hyx, psi_hzx = psix_bufs
        psi_ezy, psi_exy, psi_hzy, psi_hxy = psiy_bufs 
        psi_exz, psi_eyz, psi_hxz, psi_hyz = psiz_bufs
        pcb_half, pcb_one, pca_half, pca_one = pc_bufs

        args0 = fields.ns + [np.int32(npml)]
        arguments = {'x': {'E': {'+': args0 + [ey, ez, hz, hy, psi_eyx, psi_ezx, pcb_half, pca_half] + ces['x'], \
                                 '-': args0 + [ey, ez, hz, hy, psi_eyx, psi_ezx, pcb_one, pca_one] + ces['x']}, \
                           'H': {'+': args0 + [hz, hy, ey, ez, psi_hzx, psi_hyx, pcb_one, pca_one] + chs['x'], \
                                 '-': args0 + [hz, hy, ey, ez, psi_hzx, psi_hyx, pcb_half, pca_half] + chs['x']}}, \
                     'y': {'E': {'+': args0 + [ez, ex, hx, hz, psi_ezy, psi_exy, pcb_half, pca_half] + ces['y'], \
                                 '-': args0 + [ez, ex, hx, hz, psi_ezy, psi_exy, pcb_one, pca_one] + ces['y']}, \
                           'H': {'+': args0 + [hx, hz, ez, ex, psi_hxy, psi_hzy, pcb_one, pca_one] + chs['y'], \
                                 '-': args0 + [hx, hz, ez, ex, psi_hxy, psi_hzy, pcb_half, pca_half] + chs['y']}}, \
                     'z': {'E': {'+': args0 + [ex, ey, hy, hx, psi_exz, psi_eyz, pcb_half, pca_half] + ces['z'], \
                                 '-': args0 + [ex, ey, hy, hx, psi_exz, psi_eyz, pcb_one, pca_one] + ces['z']}, \
                           'H': {'+': args0 + [hy, hx, ex, ey, psi_hyz, psi_hxz, pcb_one, pca_one] + chs['z'], \
                                 '-': args0 + [hy, hx, ex, ey, psi_hyz, psi_hxz, pcb_half, pca_half] + chs['z']}}}

        # global variables
        self.mainf = fields
        self.directions = directions
        self.psix_bufs = psix_bufs
        self.psiy_bufs = psiy_bufs
        self.psiz_bufs = psiz_bufs
        self.pc_bufs = pc_bufs
        self.programs = programs
        self.arguments = arguments

        nn = {'x': int(npml*ny*nz), 'y': int(nx*npml*nz), 'z': int(nx*ny*npml)}
        self.gs = {}
        for key, val in nn.items():
            remainder = val % fields.ls
            self.gs[key] = val if remainder == 0 else val - remainder + fields.ls 

        # append to the update list
        self.priority_type = 'pml'
        fields.append_instance(self)


    def update(self, eh):
        for xyz, pms in zip(['x', 'y', 'z'], self.directions):
            for pm in pms:
                self.programs[xyz][eh][pm].update(self.mainf.queue, (self.gs[xyz],), (self.mainf.ls,), *self.arguments[xyz][eh][pm])


    def update_e(self):
        self.update('E')


    def update_h(self):
        self.update('H')