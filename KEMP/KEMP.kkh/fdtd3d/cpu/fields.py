# Author  : Ki-Hwan Kim
# Purpose : Base class
# Target  : CPU using C
# Created : 2012-02-08
# Modified:

import numpy as np

from kemp.fdtd3d.util import common, common_cpu
from queue_task import QueueTask


class Fields(object):
    def __init__(self, queue_task, \
                 nx, ny, nz, \
                 precision_float='single', \
                 use_cpu_core=0):
        """
        """

        common.check_type('queue_task', queue_task, QueueTask)
        common.check_type('nx', nx, int)
        common.check_type('ny', ny, int)
        common.check_type('nz', nz, int)
        common.check_value('precision_float', precision_float, ('single', 'double'))
        common.check_type('use_cpu_core', use_cpu_core, int)

        # local variables
        ns = [nx, ny, nz]
        dtype = {'single':np.float32, 'double':np.float64}[precision_float]

        # allocations
        ehs = [np.zeros(ns, dtype) for i in range(6)]

        # common macros for C templates
        dtype_macros = ['DTYPE']
        dtype_values = {'single': ['float'],' double': ['double']}[precision_float];

        omp_macros = ['OMP ', 'SET_NUM_THREADS']
        if use_cpu_core == 0:
            omp_values = ['', '']
        elif use_cpu_core == 1:
            omp_values = ['// ', '']
        else:
            omp_values = ['', 'omp_set_num_threads(%d);' % use_cpu_core]
            
        # global variables and functions
        self.device_type = 'cpu'
        self.qtask = queue_task
        self.enqueue = queue_task.enqueue
        self.enqueue_barrier = queue_task.enqueue_barrier

        self.dx = 1.
        self.dt = 0.5
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ns = ns

        self.dtype = dtype
        self.dtype_omp_macros = dtype_macros + omp_macros
        self.dtype_omp_values = dtype_values + omp_values

        self.ehs = ehs
        self.ex, self.ey, self.ez = ehs[:3]
        self.hx, self.hy, self.hz = ehs[3:]
        
        self.ce_on, self.ch_on = False, False
        self.rd_on = False

        # update list
        self.instance_list = []
        self.append_instance = lambda instance: \
                common.append_instance(self.instance_list, instance)


    def get(self, str_f):
        common.check_type('str_f', str_f, str)
        return self.__dict__[str_f]


    def get_ces(self):
        if not self.ce_on:
            self.ces = [np.ones(self.ns, self.dtype)*0.5 for i in range(3)]
            self.cex, self.cey, self.cez = self.ces
            self.ce_on = True
        
        return self.ces


    def get_chs(self):
        if not self.ch_on:
            self.chs = [np.ones(self.ns, self.dtype)*0.5 for i in range(3)]
            self.chx, self.chy, self.chz = self.chs
            self.ch_on = True
        
        return self.chs
        
    
    def get_rds(self):
        if not self.rd_on:
            self.erds = [np.ones(n, self.dtype) for n in self.ns]  # reciprocal ds
            self.hrds = [np.ones(n, self.dtype) for n in self.ns]
            for erd in self.erds: erd[-1] = 0
            for hrd in self.hrds: erd[0] = 0
            self.rd_on = True
            
        return self.erds, self.hrds
            
    
    def update_e(self):
        for instance in self.instance_list:
            instance.update_e()


    def update_h(self):
        for instance in self.instance_list:
            instance.update_h()
