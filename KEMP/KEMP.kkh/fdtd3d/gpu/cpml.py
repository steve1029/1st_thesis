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

        # psi allocations
        psi_shapes = {'x': (npml, ny, nz), 'y': (nx, npml, nz), 'z': (nx, ny, npml)}
        psi_bufs = {}
        for xyz, pms in zip(['x', 'y', 'z'], directions):
            psi_bufs[xyz] = {}
            for pm in pms:
                psi_bufs[xyz][pm] = {}
                for eh in ['E', 'H']:
                    f = np.zeros(psi_shapes[xyz], dtype)
                    bufs = [cl.Buffer(context, cl.mem_flags.READ_WRITE, f.nbytes) for i in range(2)]
                    for buf in bufs: cl.enqueue_copy(queue, buf, f)
                    psi_bufs[xyz][pm][eh] = bufs

        # coefficient allocations
        sigma = lambda i: sigma_max * (i / npml) ** m_sigma
        kappa = lambda i: 1. + (kappa_max - 1.) * (i / npml) ** m_sigma
        alpha = lambda i: alpha_max * ((npml - i) / npml) ** m_alpha
        pcb = lambda i: np.exp(-(sigma(i) / kappa(i) + alpha(i)) * dt)
        pca = lambda i: sigma(i) / ((sigma(i) + alpha(i) * kappa(i))*kappa(i)) * (pcb(i) - 1)

        i_half = np.arange(0.5, npml, 1, dtype)
        i_full = np.arange(1, npml+1, 1, dtype)
        iis = {'+': {'E': i_half, 'H': i_full}, \
               '-': {'E': i_full[::-1], 'H': i_half[::-1]}} 
        
        pc_bufs = {}
        for pm in ['+', '-']:
            pc_bufs[pm] = {}
            for eh in ['E', 'H']:
                f = np.zeros(npml, dtype)
                bufs = [cl.Buffer(context, cl.mem_flags.READ_ONLY, f.nbytes) for i in range(2)]
                for buf, pc in zip(bufs, [pcb, pca]):
                    cl.enqueue_copy(queue, buf, pc(iis[pm][eh]))
                pc_bufs[pm][eh] = bufs
                    
        # modify reciprocal ds
        if kappa_max != 1:
            sls = {'+': {'E': slice(-npml-1, -1), 'H': slice(-npml, None)}, \
                   '-': {'E': slice(None, npml), 'H': slice(1, npml+1)}} 
            for pms, erd, hrd in zip(directions, *fields.get_rds()):
                for pm in pms:
                    erd[sls[pm]['E']] /= kappa(iis[pm]['E'])
                    hrd[sls[pm]['H']] /= kappa(iis[pm]['H'])
        
        # program
        nmax = {'x': 'npml*ny*nz', 'y': 'nx*npml*nz', 'z': 'nx*ny*npml'}
        idx_pc = {'x': 'idx/(ny*nz)', 'y': '(idx/nz)%npml', 'z': 'idx%npml'}
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
        for xyz, pms in zip(['x', 'y', 'z'], directions):
            programs[xyz] = {}
            for pm in pms:
                programs[xyz][pm] = {}
                for eh in ['E', 'H']:
                    macros = ['NMAX', 'IDX_PC', 'IDX_F1', 'IDX_F2', 'IDX_F3', 'ARGS_CF', 'CF1', 'CF2', 'DTYPE', 'PRAGMA_fp64']
                    values = [nmax[xyz], idx_pc[xyz], idx_f1[eh][xyz][pm], idx_f2[eh][xyz], idx_f3[eh][xyz]]
                    if (eh == 'E' and fields.ce_on) or (eh == 'H' and fields.ch_on):
                        values += values_array
                    else:
                        values += values_const
                    values += fields.dtype_str_list
                    ksrc = common.replace_template_code(open(common_gpu.src_path + 'cpml.cl').read(), macros, values)
                    programs[xyz][pm][eh] = cl.Program(context, ksrc).build()

        # arguments
        ex, ey, ez, hx, hy, hz = fields.eh_bufs
        f_bufs = {'x': {'E': [ey, ez, hz, hy], 'H': [hz, hy, ey, ez]}, \
                  'y': {'E': [ez, ex, hx, hz], 'H': [hx, hz, ez, ex]}, \
                  'z': {'E': [ex, ey, hy, hx], 'H': [hy, hx, ex, ey]}}
        
        cf_bufs = {'x': {'E': [], 'H': []}, \
                   'y': {'E': [], 'H': []}, \
                   'z': {'E': [], 'H': []}}
        if fields.ce_on:
            cex, cey, cez = fields.ce_bufs
            cf_bufs['x']['E'] = [cey, cez]
            cf_bufs['y']['E'] = [cez, cex]
            cf_bufs['z']['E'] = [cex, cey]
        if fields.ch_on:
            chx, chy, chz = fields.ch_bufs
            cf_bufs['x']['H'] = [chz, chy]
            cf_bufs['y']['H'] = [chx, chz]
            cf_bufs['z']['H'] = [chy, chx]
        
        arguments = {}
        for xyz, pms in zip(['x', 'y', 'z'], directions):
            arguments[xyz] = {}
            for pm in pms:
                arguments[xyz][pm] = {}
                for eh in ['E', 'H']:
                    arguments[xyz][pm][eh] = fields.ns + [np.int32(npml)] + \
                        f_bufs[xyz][eh] + psi_bufs[xyz][pm][eh] + pc_bufs[pm][eh] + cf_bufs[xyz][eh]

        # global variables
        self.mainf = fields
        self.directions = directions
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
                self.programs[xyz][pm][eh].update(self.mainf.queue, (self.gs[xyz],), (self.mainf.ls,), *self.arguments[xyz][pm][eh])


    def update_e(self):
        self.update('E')


    def update_h(self):
        self.update('H')
