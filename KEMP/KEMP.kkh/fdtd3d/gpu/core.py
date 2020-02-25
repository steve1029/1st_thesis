# Author  : Ki-Hwan Kim
# Purpose : Update function for the core
# Target  : GPU using PyOpenCL
# Created : 2012-01-30
# Modified: 

import numpy as np
import pyopencl as cl

from kemp.fdtd3d.util import common, common_gpu
from fields import Fields


class Core:
    def __init__(self, fields):
        """
        """

        common.check_type('fields', fields, Fields)

        # local variables
        context = fields.context
        queue = fields.queue

        precision_float = fields.precision_float
        dtype_str_list = fields.dtype_str_list

        ce_on = fields.ce_on
        ch_on = fields.ch_on
        rd_on = fields.rd_on

        # allocation and memcpy
        if ce_on:
            for c_buf, c in zip(fields.ce_bufs, fields.ces): cl.enqueue_copy(queue, c_buf, c)
            
        if ch_on:
            for c_buf, c in zip(fields.ch_bufs, fields.chs): cl.enqueue_copy(queue, c_buf, c)

        if rd_on:
            for rd_buf, rd in zip(fields.erd_bufs, fields.erds): cl.enqueue_copy(queue, rd_buf, rd)
            for rd_buf, rd in zip(fields.hrd_bufs, fields.hrds): cl.enqueue_copy(queue, rd_buf, rd)
    
        # program
        macros = ['ARGS_CE', 'CEX', 'CEY', 'CEZ', \
                  'ARGS_CH', 'CHX', 'CHY', 'CHZ', \
                  'ARGS_RD', 'RDX', 'RDY', 'RDZ', \
                  'DX', 'DTYPE', 'PRAGMA_fp64']

        values = ['', '0.5', '0.5', '0.5', \
                  '', '0.5', '0.5', '0.5', \
                  '', '', '', '', \
                  str(fields.ls)] + dtype_str_list

        if ce_on:
            values[:4] = [ \
                ', __global DTYPE *cex, __global DTYPE *cey, __global DTYPE *cez', \
                'cex[idx]', 'cey[idx]', 'cez[idx]']

        if ch_on:
            values[4:8] = [ \
                ', __global DTYPE *chx, __global DTYPE *chy, __global DTYPE *chz', \
                'chx[idx]', 'chy[idx]', 'chz[idx]']

        if rd_on:
            values[8:12] = [ \
                ', __constant DTYPE *rdx, __constant DTYPE *rdy, __constant DTYPE *rdz', \
                'rdx[i] *', 'rdy[j] *', 'rdz[k] *']
            
        ksrc = common.replace_template_code( \
                open(common_gpu.src_path + 'core.cl').read(), macros, values)
        program = cl.Program(context, ksrc).build()

        # arguments
        e_args = fields.ns + fields.eh_bufs
        h_args = fields.ns + fields.eh_bufs
        if ce_on: e_args += fields.ce_bufs
        if ch_on: h_args += fields.ch_bufs
        if rd_on:
            e_args += fields.erd_bufs
            h_args += fields.hrd_bufs

        # global variables and functions
        self.mainf = fields
        self.program = program
        self.e_args = e_args
        self.h_args = h_args

        nx, ny, nz = fields.ns
        nmax = int( (nx * ny -1) * nz )
        remainder = nmax % fields.ls
        self.gs = nmax if remainder == 0 else nmax - remainder + fields.ls 

        # append to the update list
        self.priority_type = 'core'
        self.mainf.append_instance(self)


    def update_e(self):
        self.program.update_e(self.mainf.queue, (self.gs,), (self.mainf.ls,), *self.e_args)


    def update_h(self):
        self.program.update_h(self.mainf.queue, (self.gs,), (self.mainf.ls,), *self.h_args)
