# Author  : Ki-Hwan Kim
# Purpose : Update function for the core
# Target  : CPU using C
# Created : 2012-02-08
# Modified: 

import numpy as np

from kemp.fdtd3d.util import common, common_cpu
from fields import Fields


class Core:
    def __init__(self, fields):
        """
        """

        common.check_type('fields', fields, Fields)

        # local variables
        dtype = fields.dtype
        nx, ny, nz = ns = fields.ns

        ce_on = fields.ce_on
        ch_on = fields.ch_on
        rd_on = fields.rd_on

        # program
        cf_values = {True: ['', 'OOOOOOOOO', ', &Cx, &Cy, &Cz', 'cx[idx]', 'cy[idx]', 'cz[idx]'], \
                     False: ['// ', 'OOOOOO', '', '0.5', '0.5', '0.5']}
        ce_macros = ['COEFF_E ', 'PARSE_ARGS_CE', 'PYARRAYOBJECT_CE', 'CEX', 'CEY', 'CEZ']
        ce_values = cf_values[ce_on]
        ch_macros = ['COEFF_H ', 'PARSE_ARGS_CH', 'PYARRAYOBJECT_CH', 'CHX', 'CHY', 'CHZ']
        ch_values = cf_values[ch_on]
        
        macros = fields.dtype_omp_macros + ce_macros + ch_macros
        values = fields.dtype_omp_values + ce_values + ch_values

        ksrc = common.replace_template_code( \
                open(common_cpu.src_path + 'core.c').read(), macros, values)
        program = common_cpu.build_clib(ksrc, 'core')

        # arguments
        e_args = fields.ehs
        h_args = fields.ehs
        if ce_on:
            e_args += fields.ces
        if ch_on:
            h_args += fields.chs

        # global variables
        self.mainf = fields
        self.program = program
        self.e_args = e_args
        self.h_args = h_args

        # append to the update list
        self.priority_type = 'core'
        fields.append_instance(self)


    def update_e(self):
        self.mainf.enqueue(self.program.update_e, self.e_args)


    def update_h(self):
        self.mainf.enqueue(self.program.update_h, self.h_args)