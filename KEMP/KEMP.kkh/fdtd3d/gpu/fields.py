# Author  : Ki-Hwan Kim
# Purpose : Base class
# Target  : GPU using PyOpenCL
# Created : 2011-11-01
# Modified: 2012-01-30  dynamic coefficients (ce, ch, rd)

import numpy as np
import pyopencl as cl

from kemp.fdtd3d.util import common, common_gpu


class Fields:
    def __init__(self, context, device, \
                 nx, ny, nz, \
                 precision_float='single', \
                 local_work_size=256):
        """
        """

        common.check_type('context', context, cl.Context)
        common.check_type('device', device, cl.Device)
        common.check_type('nx', nx, int)
        common.check_type('ny', ny, int)
        common.check_type('nz', nz, int)
        common.check_value('precision_float', precision_float, ('single', 'double'))
        common.check_type('local_work_size', local_work_size, int)

        # local variables
        ns = [np.int32(nx), np.int32(ny), np.int32(nz)]

        queue = cl.CommandQueue(context, device)
        pragma_fp64 = ''
        if precision_float == 'double':
            extensions = device.get_info(cl.device_info.EXTENSIONS)
            if 'cl_khr_fp64' in extensions:
                pragma_fp64 = '#pragma OPENCL EXTENSION cl_khr_fp64 : enable'
            elif 'cl_amd_fp64' in extensions:
                pragma_fp64 = '#pragma OPENCL EXTENSION cl_amd_fp64 : enable'
            else:
                precision_float = 'single'
                print('Warning: The %s GPU device is not support the double-precision.') % \
                        device.get_info(cl.device_info.NAME)
                print('The precision is changed to \'single\'.')

        dtype = {'single':np.float32, 'double':np.float64}[precision_float]
        dtype_str_list = { \
                'single':['float', ''], \
                'double':['double', pragma_fp64] }[precision_float]

        # allocations
        f = np.zeros(ns, dtype)
        eh_bufs = [cl.Buffer(context, cl.mem_flags.READ_WRITE, f.nbytes) for i in range(6)]
        for eh_buf in eh_bufs: cl.enqueue_copy(queue, eh_buf, f)
        
        # global variables
        self.device_type = 'gpu'
        self.context = context
        self.device = device
        self.queue = queue

        self.dx = 1.
        self.dt = 0.5
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ns = ns

        self.precision_float = precision_float
        self.dtype = dtype
        self.dtype_str_list = dtype_str_list 

        self.eh_bufs = eh_bufs
        self.ex_buf, self.ey_buf, self.ez_buf = eh_bufs[:3]
        self.hx_buf, self.hy_buf, self.hz_buf = eh_bufs[3:]
        
        self.ce_on, self.ch_on = False, False
        self.rd_on = False
        
        self.ls = local_work_size

        # create update list
        self.instance_list = []
        self.append_instance = lambda instance: \
            common.append_instance(self.instance_list, instance)


    def get_buf(self, str_f):
        common.check_type('str_f', str_f, str)
        return self.__dict__[str_f + '_buf']


    def get_ces(self):
        if not self.ce_on:
            self.ces = [np.ones(self.ns, self.dtype)*0.5 for i in range(3)]
            self.cex, self.cey, self.cez = self.ces
            self.ce_bufs = [cl.Buffer(self.context, cl.mem_flags.READ_ONLY, ce.nbytes) for ce in self.ces]
            self.ce_on = True
        
        return self.ces
        
        
    def get_chs(self):
        if not self.ch_on:
            self.chs = [np.ones(self.ns, self.dtype)*0.5 for i in range(3)]
            self.chx, self.chy, self.chz = self.chs
            self.ch_bufs = [cl.Buffer(self.context, cl.mem_flags.READ_ONLY, ch.nbytes) for ch in self.chs]
            self.ch_on = True
        
        return self.chs
        

    def get_rds(self):
        if not self.rd_on:
            self.erds = [np.ones(n, self.dtype) for n in self.ns]  # reciprocal ds
            self.hrds = [np.ones(n, self.dtype) for n in self.ns]
            for erd in self.erds: erd[-1] = 0
            for hrd in self.hrds: erd[0] = 0
            self.erd_bufs = [cl.Buffer(self.context, cl.mem_flags.READ_ONLY, rd.nbytes) for rd in self.erds]
            self.hrd_bufs = [cl.Buffer(self.context, cl.mem_flags.READ_ONLY, rd.nbytes) for rd in self.hrds]
            self.rd_on = True
            
        return self.erds, self.hrds
            
    
    def update_e(self):
        for instance in self.instance_list:
            instance.update_e()


    def update_h(self):
        for instance in self.instance_list:
            instance.update_h()
