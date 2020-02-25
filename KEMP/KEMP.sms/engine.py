import os
import numpy as np
import threading
import Queue
import atexit
import tempfile
import subprocess
import multiprocessing as mproc

from ctypes import c_int, c_float, c_double
from datetime import datetime as dtm

import exception
from util import *

class engine:
    def __init__(self):
        self.templates   = {}
        self.programs    = {}
        self.priority_order_e = ['pole_e', 'main_e', 'disp_e', \
                                 'pml_e', 'pbc_e', 'src_e', 'tfsf_e', 'rft_e']
        self.priority_order_h = ['pole_h', 'main_h', 'disp_h', \
                                 'pml_h', 'pbc_h', 'src_h', 'tfsf_h', 'rft_h']
        self.updates = {'main_e':None, 'main_h':None, \
                        'pole_e':None, 'pole_h':None, \
                        'disp_e':None, 'disp_h':None, \
                        'pml_e' :None, 'pml_h' :None, \
                        'pbc_e' :None, 'pbc_h' :None, \
                        'tfsf_e':None, 'tfsf_h':None, \
                        'rft_e' :None, 'rft_h' :None}
        self.kernels = {}
        self.kernel_args = {}

class subengine(engine):
    def __init__(self, main_engine):
        engine.__init__(self)
        self.main_engine = main_engine
        self.templates = main_engine.templates
        if not 'cpu' in main_engine.name:
            self.programs['get_set_data'] = main_engine.programs['get_set_data']
            self.programs['src'] = main_engine.programs['src']
            self.kernels['src_a'] = main_engine.kernels['src_a']
            self.kernels['src_v'] = main_engine.kernels['src_v']
            self.kernels['get_data'] = main_engine.kernels['get_data']
            self.kernels['set_data_values'] = main_engine.kernels['set_data_values']
            self.kernels['set_data_svalue'] = main_engine.kernels['set_data_svalue']

    def __getattr__(self, name):
        return getattr(self.main_engine, name)

class intel_cpu(engine):
    def __init__(self):
        '''
        intel_cpu class is a computing engine.
        '''
        engine.__init__(self)
        self.name  = 'intel_cpu'
        self.queue = None
        self.ctx   = None
        self.cmd_queue = mproc.Queue()
        self.out_queue = mproc.Queue()
        self.err_queue = mproc.Queue()
        self.code_path = tempfile.mkdtemp()
        self.clib_path = tempfile.mkdtemp()
        self.code_num  = 0
        self.src_path  = os.path.dirname(os.path.abspath(__file__)) + '/src/cpu/'
#        self.cmd       = '/opt/intel/bin/icc -ansi-alias -fPIC -shared -O3 -xSSE3 -openmp -liomp5 -g -std=c99 %s -o %s'
        self.cmd       = '/opt/intel/bin/icc -ansi-alias -fPIC -shared -O3 -xAVX -openmp -g -std=c99 %s -o %s'
#        self.cmd       = '/opt/intel/bin/icc -static-intel -ansi-alias -fPIC -shared -O3 -xSSE3 -g -std=c99 %s -o %s'
        self.executor  = Ext_executor(self.cmd_queue, self.out_queue, self.err_queue)
        self.executor.start()
        atexit.register(self.cmd_queue.put, None)

    def build(self, code):
        self.code_num += 1
        code_path = self.code_path + ('/%s.c' % self.code_num)
        f = open(code_path, 'w')
        f.write(code)
        f.close()
        cmd = self.cmd % (code_path, self.clib_path + ('/%s.so' % self.code_num))
#        cmd_proc = subprocess.Popen(cmd.split(), \
#                                    stdout=subprocess.PIPE, \
#                                    stderr=subprocess.PIPE)
#        out, err = cmd_proc.communicate()
        self.cmd_queue.put(cmd)
        out = self.out_queue.get()
        err = self.err_queue.get()
        if err != '':
            if ('err' in err) or ('Err' in err):
                msg_err  = file('./.msg_err.txt', 'w')
                msg_code = file('./.msg_code.c', 'w')
                msg_err.write(err)
                msg_code.write(code)
                msg_err.close()
                msg_code.close()
                raise exception.ComputingEngineError, '%s\n%s\n%s' % (err, 'The error code is', code)
#            else:
#                print err
        codename = '%s' % self.code_num
        clib =  np.ctypeslib.load_library(codename, self.clib_path)
        return clib

    def arg_to_arg_types(self, args):
        arg_types = []
        for arg in args[3:]:
            if   isinstance(arg, np.int32):
                arg_type = c_int
            elif isinstance(arg, np.float32):
                arg_type = c_float
            elif isinstance(arg, np.float64):
                arg_type = c_double
            elif isinstance(arg, np.ndarray):
                arg_type = np.ctypeslib.ndpointer(dtype = arg.dtype, \
                                                  ndim  = arg.ndim, \
                                                  shape = np.shape(arg), \
                                                  flags = 'C_CONTIGUOUS, ALIGNED')
            else:
                raise TypeError, 'Invaild arg types... arg types must be in [numpy.int32, numpy.float32, numpy.float64]'
            arg_types.append(arg_type)
        return arg_types

    def set_kernel(self, kernel, args):
        arg_types = self.arg_to_arg_types(args)
        kernel.restype  = None
        kernel.argtypes = arg_types
        return kernel

    def test_init(self):
        self.dtype = np.float32
        self.device_id  = 0
        self.is_subfdtd = False

        self.block = (256, 1, 1)
        try:
            import psutil
            self.ls    = psutil.NUM_CPUS
            self.total_memory_size = psutil.TOTAL_PHYMEM
        except ImportError:
            import multiprocessing
            self.ls    = multiprocessing.cpu_count()
        self.gs    = 10000

        self.code_dtype = {np.float32:'float', np.float64:'double'}[comp_to_real(self.dtype)]
        self.code_prev  = ['__GET_GLOBAL_INDEX__', \
                           '__GET_LOCAL_INDEX__', \
                           '__GLOBAL_BARRIER__', \
                           '__LOCAL_BARRIER__', \
                           '__KERNEL__', '__DEVICE__', \
                           '__GLOBAL__', '__LOCAL__', '__LOCAL_SIZE__', \
                           '__ACTIVATE_FLOAT64__', '__CONSTANT__', '__FLOAT__']
        self.code_post  = ['', \
                           '', \
                           '', \
                           '', \
                           '', '', \
                           '', '', str(self.ls), \
                           '', '', self.code_dtype]

    def init(self, fdtd):
        self.fdtd = fdtd
        self.dtype = fdtd.dtype
        self.device_id  = fdtd.device_id
        self.is_subfdtd = fdtd.is_subfdtd

        self.allocated_mem_size = 0

        try:
            import psutil
            self.ls    = psutil.NUM_CPUS
            self.total_memory_size = psutil.TOTAL_PHYMEM
            self.max_mem_size = psutil.TOTAL_PHYMEM
        except ImportError:
            import multiprocessing
            self.ls    = multiprocessing.cpu_count()
        self.block = (256, 1, 1)
        self.gs    = fdtd.n

        self.src_path   = os.path.dirname(os.path.abspath(__file__)) + '/src/cpu/'
        self.code_dtype = {np.float32:'float', np.float64:'double'}[comp_to_real(self.dtype)]

        self.code_prev  = ['__GET_GLOBAL_INDEX__', \
                           '__GET_LOCAL_INDEX__', \
                           '__GLOBAL_BARRIER__', \
                           '__LOCAL_BARRIER__', \
                           '__KERNEL__', '__DEVICE__', \
                           '__GLOBAL__', '__LOCAL__', '__LOCAL_SIZE__', \
                           '__ACTIVATE_FLOAT64__', '__CONSTANT__', '__FLOAT__']
        self.code_post  = ['', \
                           '', \
                           '', \
                           '', \
                           '', '', \
                           '', '', str(self.ls), \
                           '', '', self.code_dtype]

        for src_code_file in os.listdir(self.src_path):
            src_code = file('%s/%s' % (self.src_path, src_code_file), 'r').read()
            self.templates[src_code_file.replace('.c', '')] = src_code

class intel_cpu_without_openmp(intel_cpu):
    def __init__(self):
        intel_cpu.__init__(self)
        self.name = 'intel_cpu_without_openmp'
#        self.cmd  = '/opt/intel/bin/icc  -ansi-alias -fPIC -shared -O3 -xSSE3 -g -std=c99 %s -o %s'
        self.cmd  = '/opt/intel/bin/icc -ansi-alias -fPIC -shared -O3 -xAVX -g -std=c99 %s -o %s'

class gnu_cpu(intel_cpu):
    def __init__(self):
        intel_cpu.__init__(self)
        self.name = 'intel_cpu'
        self.cmd   = 'gcc -O3 -Wall -fPIC -msse -fopenmp -std=c99 -shared %s -o %s'

class vc_cpu(intel_cpu):
    def __init__(self):
        intel_cpu.__init__(self)
        self.name = 'intel_cpu'
        self.cmd   = 'cl -static-intel -ansi-alias -fPIC -shared -O3 -xSSE3 -g -std=c99 %s -o %s'

class nvidia_cuda(engine):
    def __init__(self):
        '''
        nvidia_cuda class is a computing engine, NO thread.
        '''
        engine.__init__(self)
        self.name = 'nvidia_cuda'
        try:
            import pycuda
        except ImportError:
            raise ImportError, 'pycuda module is needed for using ComputingEngine \'nvidia_cuda\''
        import pycuda.driver as driver
        from pycuda.compiler import SourceModule

        self.src_path   = os.path.dirname(os.path.abspath(__file__)) + '/src/nvidia/'

        self.cuda = pycuda
        self.drv  = driver
        self.SourceModule = SourceModule
        self.drv.init()
        self.queue  = None

    def get_function(self, mod, name):
        self.ctx.push()
        return mod.get_function(name)

    def prepare(self, cuda_func, args):
        self.ctx.push()
        arg_types = []
        for arg in args[3:]:
            if   isinstance(arg, np.int32):
                arg_type = np.int32
            elif isinstance(arg, np.float32):
                arg_type = np.float32
            elif isinstance(arg, np.float64):
                arg_type = np.float64
            elif isinstance(arg, self.drv.DeviceAllocation):
                arg_type = self.drv.DeviceAllocation
            else:
                raise exception.ComputingEngineError, 'nVIDIA CUDA kernel argument error: Invalid type'
            arg_types.append(arg_type)
        cuda_func.prepare(arg_types)

    def get_global(self, mod, name):
        self.ctx.push()
        return mod.get_global(name)

    def test_init(self):
        self.dtype = np.float32
        self.device_id  = 0
        self.is_subfdtd = False

        self.block = (256, 1, 1)
        self.ls    = self.block[0]*self.block[1]*self.block[2]
        self.gs    = 10000

        self.device = self.drv.Device(self.device_id)
        self.ctx    = self.device.make_context()
        self.grid   = cuda_grid(self.gs, self.ls)

        atexit.register(self.ctx.detach , *[])
        atexit.register(self.ctx.pop    , *[])
        atexit.register(self.ctx.push   , *[])

        self.code_dtype = {np.float32:'float', np.float64:'double'}[comp_to_real(self.dtype)]
        self.code_prev  = ['__GET_GLOBAL_INDEX__', \
                           '__GET_LOCAL_INDEX__', \
                           '__GLOBAL_BARRIER__', \
                           '__LOCAL_BARRIER__', \
                           '__KERNEL__', '__DEVICE__', \
                           '__GLOBAL__', '__LOCAL__', '__LOCAL_SIZE__', \
                           '__ACTIVATE_FLOAT64__', '__CONSTANT__', '__FLOAT__']
        self.code_post  = ['gridDim.y*blockDim.x*blockIdx.x + blockDim.x*blockIdx.y + threadIdx.x', \
                           'threadIdx.x', \
                           '', \
                           '__syncthreads()', \
                           '__global__', '__device__', \
                           '', '__shared__', str(self.ls), \
                           '', '__constant__', self.code_dtype]

        # Program of controlling fields in this FDTD_world instances
        for src_code_file in os.listdir(self.src_path):
            src_code = file('%s/%s' % (self.src_path, src_code_file), 'r').read()
            self.templates[src_code_file] = src_code

        """
        code_prev = ['__DECLARE_CONSTANT_ARRAYS__', \
                     'strts[i]', 'steps[i]', 'd_dev[i]', 'd_buf[i]']
        code_post = ['''__DEVICE__ __CONSTANT__ int cst_strts[__LOCAL_SIZE__];
                        __DEVICE__ __CONSTANT__ int cst_steps[__LOCAL_SIZE__];
                        __DEVICE__ __CONSTANT__ int cst_d_dev[__LOCAL_SIZE__];
                        __DEVICE__ __CONSTANT__ int cst_d_buf[__LOCAL_SIZE__];''', \
                       'cst_strts[i]', 'cst_steps[i]', 'cst_d_dev[i]', 'cst_d_buf[i]']
        """
        code_prev = ['__DECLARE_SHARED_ARRAYS__', \
                     'strts[i]', 'steps[i]', 'd_dev[i]', 'd_buf[i]']
        code_post = [ \
'''
    __LOCAL__ int cst_strts[__LOCAL_SIZE__];
    __LOCAL__ int cst_steps[__LOCAL_SIZE__];
    __LOCAL__ int cst_d_dev[__LOCAL_SIZE__];
    __LOCAL__ int cst_d_buf[__LOCAL_SIZE__];
    int tx = __GET_LOCAL_INDEX__;
    if(tx<ndim)
    {
        cst_strts[tx] = strts[tx];
        cst_steps[tx] = steps[tx];
        cst_d_dev[tx] = d_dev[tx];
        cst_d_buf[tx] = d_buf[tx];
    }
    __LOCAL_BARRIER__;
''', \
                       'cst_strts[i]', 'cst_steps[i]', 'cst_d_dev[i]', 'cst_d_buf[i]']
        code = template_to_code(self.templates['get_set_data'], code_prev+self.code_prev, code_post+self.code_post)
        self.programs['get_set_data'] = self.build(code)
        self.kernels['get_data']        = self.get_function(self.programs['get_set_data'], 'get_data')
        self.kernels['set_data_svalue'] = self.get_function(self.programs['get_set_data'], 'set_data_svalue')
        self.kernels['set_data_values'] = self.get_function(self.programs['get_set_data'], 'set_data_values')
        self.kernels['get_data'].prepare([np.int32, np.int32, \
                                          self.drv.DeviceAllocation, \
                                          self.drv.DeviceAllocation, \
                                          self.drv.DeviceAllocation, \
                                          self.drv.DeviceAllocation, \
                                          self.drv.DeviceAllocation, \
                                          self.drv.DeviceAllocation] )
        self.kernels['set_data_values'].prepare([np.int32, np.int32, np.int32, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation] )
        self.kernels['set_data_svalue'].prepare([np.int32, np.int32, np.int32, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 comp_to_real(self.fdtd.dtype)] )

        code = template_to_code(fdtd.engine.templates['incident_direct'], fdtd.engine.code_prev, fdtd.engine.code_post)
        self.programs['src'] = self.build(code)
        self.kernels['src_a'] = self.get_function(self.programs['src'], 'update_from_array')
        self.kernels['src_v'] = self.get_function(self.programs['src'], 'update_from_value')
        self.kernels['src_m'] = self.get_function(self.programs['src'], 'update_mnch')
        self.kernels['src_a'].prepare([np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       self.drv.DeviceAllocation, self.drv.DeviceAllocation])
        self.kernels['src_v'].prepare([np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       self.drv.DeviceAllocation, self.dtype])
        self.kernels['src_m'].prepare([np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       self.dtype, self.dtype, \
                                       self.drv.DeviceAllocation, \
                                       self.drv.DeviceAllocation, \
                                       self.drv.DeviceAllocation] )

        code = template_to_code(fdtd.engine.templates['running_fourier_transform'], fdtd.engine.code_prev, fdtd.engine.code_post)
        self.programs['rft'] = self.build(code)
        self.kernels['rft'] = self.get_function(self.programs['rft'], 'update')
        self.kernels['rft'].prepare([np.int32, np.int32, np.int32, \
                                     np.int32, np.int32, np.int32, \
                                     np.int32, np.int32, np.int32, \
                                     np.int32, np.int32, \
                                     comp_to_real(fdtd.dtype), \
                                     comp_to_real(fdtd.dtype), \
                                     comp_to_real(fdtd.dtype), \
                                     self.drv.DeviceAllocation, \
                                     self.drv.DeviceAllocation, \
                                     self.drv.DeviceAllocation, \
                                     self.drv.DeviceAllocation] )

    def init(self, fdtd):
        self.fdtd = fdtd
        self.dtype = fdtd.dtype
        self.device_id  = fdtd.device_id
        self.is_subfdtd = fdtd.is_subfdtd

#        self.block = (32, 16, 1)
        self.block = (32, 16, 1)
        self.ls    = self.block[0]*self.block[1]*self.block[2]
        self.gs    = fdtd.n

        self.device = self.drv.Device(self.device_id)
        self.ctx    = self.device.make_context()
        self.stream = self.drv.Stream()
        self.grid   = cuda_grid(self.gs, self.ls)

        self.allocated_mem_size = 0
        self.max_mem_size = self.device.total_memory()

        atexit.register(self.ctx.detach, *[])
        atexit.register(self.ctx.pop   , *[])
        atexit.register(self.ctx.push  , *[])

        self.code_dtype = {np.float32:'float', np.float64:'double'}[comp_to_real(self.dtype)]
        self.code_prev  = ['__GET_GLOBAL_INDEX__', \
                           '__GET_LOCAL_INDEX__', \
                           '__GLOBAL_BARRIER__', \
                           '__LOCAL_BARRIER__', \
                           '__KERNEL__', '__DEVICE__', \
                           '__GLOBAL__', '__LOCAL__', '__LOCAL_SIZE__', \
                           '__ACTIVATE_FLOAT64__', '__CONSTANT__', '__FLOAT__']
        '''                
        self.code_post  = ['gridDim.y*blockDim.x*blockIdx.x + blockDim.x*blockIdx.y + threadIdx.x', \
                           'threadIdx.x', \
                           '', \
                           '__syncthreads()', \
                           '__global__', '__device__', \
                           '', '__shared__', str(self.ls), \
                           '', '__constant__', self.code_dtype]
        '''
        self.code_post  = [cuda_get_global_index_2d_3d, \
                           cuda_get_local_index_2d_3d, \
                           '', \
                           '__syncthreads()', \
                           '__global__', '__device__', \
                           '', '__shared__', str(self.ls), \
                           '', '__constant__', self.code_dtype]
        # Program of controlling fields in this FDTD_world instances
        for src_code_file in os.listdir(self.src_path):
            src_code = file('%s/%s' % (self.src_path, src_code_file), 'r').read()
            self.templates[src_code_file] = src_code

        """
        code_prev = ['__DECLARE_CONSTANT_ARRAYS__', \
                     'strts[i]', 'steps[i]', 'd_dev[i]', 'd_buf[i]']
        code_post = ['''__DEVICE__ __CONSTANT__ int cst_strts[__LOCAL_SIZE__];
                        __DEVICE__ __CONSTANT__ int cst_steps[__LOCAL_SIZE__];
                        __DEVICE__ __CONSTANT__ int cst_d_dev[__LOCAL_SIZE__];
                        __DEVICE__ __CONSTANT__ int cst_d_buf[__LOCAL_SIZE__];''', \
                       'cst_strts[i]', 'cst_steps[i]', 'cst_d_dev[i]', 'cst_d_buf[i]']
        """
        code_prev = ['__DECLARE_SHARED_ARRAYS__', \
                     'strts[i]', 'steps[i]', 'd_dev[i]', 'd_buf[i]']
        code_post = [ \
'''
    __LOCAL__ int cst_strts[__LOCAL_SIZE__];
    __LOCAL__ int cst_steps[__LOCAL_SIZE__];
    __LOCAL__ int cst_d_dev[__LOCAL_SIZE__];
    __LOCAL__ int cst_d_buf[__LOCAL_SIZE__];
    int tx = __GET_LOCAL_INDEX__;
    if(tx<ndim)
    {
        cst_strts[tx] = strts[tx];
        cst_steps[tx] = steps[tx];
        cst_d_dev[tx] = d_dev[tx];
        cst_d_buf[tx] = d_buf[tx];
    }
    __LOCAL_BARRIER__;
''', \
                       'cst_strts[i]', 'cst_steps[i]', 'cst_d_dev[i]', 'cst_d_buf[i]']
        code = template_to_code(self.templates['get_set_data'], code_prev+self.code_prev, code_post+self.code_post)
        self.programs['get_set_data'] = self.build(code)
        self.kernels['get_data']        = self.get_function(self.programs['get_set_data'], 'get_data')
        self.kernels['set_data_svalue'] = self.get_function(self.programs['get_set_data'], 'set_data_svalue')
        self.kernels['set_data_values'] = self.get_function(self.programs['get_set_data'], 'set_data_values')
        self.kernels['get_data'].prepare([np.int32, np.int32, \
                                          self.drv.DeviceAllocation, \
                                          self.drv.DeviceAllocation, \
                                          self.drv.DeviceAllocation, \
                                          self.drv.DeviceAllocation, \
                                          self.drv.DeviceAllocation, \
                                          self.drv.DeviceAllocation] )
        self.kernels['set_data_values'].prepare([np.int32, np.int32, np.int32, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation] )
        self.kernels['set_data_svalue'].prepare([np.int32, np.int32, np.int32, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 self.drv.DeviceAllocation, \
                                                 comp_to_real(self.fdtd.dtype)] )

        code = template_to_code(fdtd.engine.templates['incident_direct'], fdtd.engine.code_prev, fdtd.engine.code_post)
        self.programs['src'] = self.build(code)
        self.kernels['src_a'] = self.get_function(self.programs['src'], 'update_from_array')
        self.kernels['src_v'] = self.get_function(self.programs['src'], 'update_from_value')
        self.kernels['src_m'] = self.get_function(self.programs['src'], 'update_mnch')
        self.kernels['src_a'].prepare([np.int32, np.int32, np.int32, \
                                       #np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       self.drv.DeviceAllocation, self.drv.DeviceAllocation])
        self.kernels['src_v'].prepare([np.int32, np.int32, np.int32, \
                                       #np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       self.drv.DeviceAllocation, comp_to_real(self.fdtd.dtype)])
        self.kernels['src_m'].prepare([np.int32, np.int32, np.int32, \
                                       #np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       np.int32, np.int32, np.int32, \
                                       comp_to_real(fdtd.dtype), \
                                       comp_to_real(fdtd.dtype), \
                                       comp_to_real(fdtd.dtype), \
                                       self.drv.DeviceAllocation, \
                                       self.drv.DeviceAllocation, \
                                       self.drv.DeviceAllocation] )

        code = template_to_code(fdtd.engine.templates['running_fourier_transform'], fdtd.engine.code_prev, fdtd.engine.code_post)
        self.programs['rft'] = self.build(code)
        self.kernels['rft'] = self.get_function(self.programs['rft'], 'update')
        self.kernels['rft'].prepare([np.int32, np.int32, np.int32, \
                                     np.int32, np.int32, np.int32, \
                                     np.int32, np.int32, np.int32, \
                                     np.int32, np.int32, \
                                     comp_to_real(fdtd.dtype), \
                                     comp_to_real(fdtd.dtype), \
                                     comp_to_real(fdtd.dtype), \
                                     self.drv.DeviceAllocation, \
                                     self.drv.DeviceAllocation, \
                                     self.drv.DeviceAllocation, \
                                     self.drv.DeviceAllocation] )

    def enqueue(self, func, args, wait=False):
        self.ctx.push()
        evt = self.drv.Event()
        evt.wait = evt.synchronize
        func(*args)
        evt.record(self.stream)
        if wait:
            evt.wait()
        return evt

    def enqueue_kernel(self, func, args, wait=False):
        gs = args[1][0]
        ls = args[2][0]
        block = self.block
        grid  = cuda_grid(gs, ls)
        self.ctx.push()
        evt = self.drv.Event()
        evt.wait = evt.synchronize
        func.prepared_async_call(grid, block, self.stream, *args[3:])
        evt.record(stream=self.stream)
        if wait:
            evt.wait()
        return evt

    def build(self, code):
        self.ctx.push()
        mod = self.SourceModule(code)
        return mod

    def get_function(self, mod, name):
        self.ctx.push()
        func = mod.get_function(name)
        return func

    def get_global(self, mod, name):
        self.ctx.push()
        var_global = mod.get_global(name)
        return var_global

    def mem_alloc(self, *args):
        self.ctx.push()
        mem_obj = self.drv.mem_alloc(*args)
        return mem_obj

    def to_device(self, *args):
        self.ctx.push()
        mem_obj = self.drv.to_device(*args)
        return mem_obj

    def from_device(self, *args):
        self.ctx.push()
        mem_obj = self.drv.from_device(*args)
        return mem_obj

class nvidia_opencl(engine):
    def __init__(self):
        engine.__init__(self)
        self.name = 'nvidia_opencl'
        try:
            import pyopencl
        except ImportError:
            raise ImportError, 'pyopencl module is needed for using ComputingEngine \'nvidia_opencl\''
        self.cl = pyopencl

        self.src_path   = os.path.dirname(os.path.abspath(__file__)) + '/src/nvidia/'

    def build(self, code):
        return self.cl.Program(self.ctx, code).build()

    def test_init(self):
        self.device_id  = 0
        self.is_subfdtd = False
        self.dtype = np.float32
        # set the fdtd environment on OpenCL context
        try:
            for platform in self.cl.get_platforms():
                if 'NVIDIA' in platform.name:
                    self.platform = platform
        except:
            import exception
            raise exception.ComputingEngineError, 'OpenCL platform \"NVIDIA CUDA\" is not exist in this system'
        self.devices  = self.platform.get_devices()
        self.device   = self.devices[self.device_id]
        self.ctx      = self.cl.Context(self.devices)

        self.ls = 256
#        self.gs = cl_global_size(fdtd.n, self.ls)
        self.n  = 10000
        self.gs = cl_global_size(self.n, self.ls)
        self.queue = self.cl.CommandQueue(self.ctx, self.device)

        # setting precision: in case of ['np.float32, np.float64']
        self.activate_float64 = ''
        if self.dtype in [np.float64, np.complex128]:
            extensions = self.device.get_info(self.cl.device_info.EXTENSIONS)
            if   'cl_khr_fp64' in extensions:
                self.activate_float64 = '#pragma OPENCL EXTENSION cl_khr_fp64 : enable'
#            elif 'cl_amd_fp64' in extensions:
#                self.activate_float64 = '#pragma OPENCL EXTENSION cl_amd_fp64 : enable'
            else:
                if self.dtype == np.complex128: self.dtype=np.complex64
                print('Warning: The %s GPU device is not support the dtype float64 and complex128.') % \
                        self.device.get_info(self.cl.device_info.NAME)
                print('The dtype is changed to float32 or complex64.')

        self.code_dtype = {np.float32:'float', np.float64:'double'}[comp_to_real(self.dtype)]
        self.code_prev  = ['__GET_GLOBAL_INDEX__', \
                           '__GET_LOCAL_INDEX__', \
                           '__GLOBAL_BARRIER__', \
                           '__LOCAL_BARRIER__', \
                           '__KERNEL__', '__DEVICE__', \
                           '__GLOBAL__', '__LOCAL__', '__LOCAL_SIZE__', \
                           '__ACTIVATE_FLOAT64__', '__CONSTANT__', '__FLOAT__']
        self.code_post  = ['get_global_id(0)', \
                           'get_local_id(0)', \
                           'barrier(CLK_GLOBAL_MEM_FENCE)', \
                           'barrier(CLK_LOCAL_MEM_FENCE)', \
                           '__kernel', '',
                           '__global', '__local', str(self.ls), \
                           self.activate_float64, '', self.code_dtype]

        for src_code_file in os.listdir(self.src_path):
            src_code = file('%s/%s' % (self.src_path, src_code_file), 'r').read()
            self.templates[src_code_file] = src_code
        # Program of controlling fields in this FDTD_world instances
#        code_prev = ['__DECLARE_CONSTANT_ARRAYS__', \
#                     'strts[i]', 'steps[i]', 'd_dev[i]', 'd_buf[i]']
        code_prev = ['__DECLARE_SHARED_ARRAYS__', \
                     'strts[i]', 'steps[i]', 'd_dev[i]', 'd_buf[i]']
        code_post = ['', \
                     'strts[i]', 'steps[i]', 'd_dev[i]', 'd_buf[i]']
        code = template_to_code(self.templates['get_set_data'], code_prev+self.code_prev, code_post+self.code_post)
        self.programs['get_set_data']  = self.cl.Program(self.ctx, code).build()

    def init(self, fdtd):
        self.fdtd = fdtd
        self.dtype = fdtd.dtype
        self.device_id  = fdtd.device_id
        self.is_subfdtd = fdtd.is_subfdtd
        # set the fdtd environment on OpenCL context
        try:
            for platform in self.cl.get_platforms():
                if 'NVIDIA' in platform.name:
                    self.platform = platform
        except:
            raise self.cl.Error, 'OpenCL platform \"NVIDIA CUDA\" is not exist in this system'
        self.devices  = self.platform.get_devices()
        self.device   = self.devices[self.device_id]
        self.ctx      = self.cl.Context(self.devices)

        self.ls = 512
        self.gs = cl_global_size(fdtd.n, self.ls)
        self.queue = self.cl.CommandQueue(self.ctx, self.device)

        self.allocated_mem_size = 0
        self.max_mem_size = self.device.max_mem_alloc_size

        # setting precision: in case of ['np.float32, np.float64']
        self.activate_float64 = ''
        if self.dtype in [np.float64, np.complex128]:
            extensions = self.device.get_info(self.cl.device_info.EXTENSIONS)
            if   'cl_khr_fp64' in extensions:
                self.activate_float64 = '#pragma OPENCL EXTENSION cl_khr_fp64 : enable'
#            elif 'cl_amd_fp64' in extensions:
#                self.activate_float64 = '#pragma OPENCL EXTENSION cl_amd_fp64 : enable'
            else:
                if self.dtype == np.complex128: self.dtype=np.complex64
                print('Warning: The %s GPU device is not support the dtype float64 and complex128.') % \
                        self.device.get_info(self.cl.device_info.NAME)
                print('The dtype is changed to float32 or complex64.')

        self.code_dtype = {np.float32:'float', np.float64:'double'}[comp_to_real(self.dtype)]
        self.code_prev  = ['__GET_GLOBAL_INDEX__', \
                           '__GET_LOCAL_INDEX__', \
                           '__GLOBAL_BARRIER__', \
                           '__LOCAL_BARRIER__', \
                           '__KERNEL__', '__DEVICE__', \
                           '__GLOBAL__', '__LOCAL__', '__LOCAL_SIZE__', \
                           '__ACTIVATE_FLOAT64__', '__CONSTANT__', '__FLOAT__']
        self.code_post  = ['get_global_id(0)', \
                           'get_local_id(0)', \
                           'barrier(CLK_GLOBAL_MEM_FENCE)', \
                           'barrier(CLK_LOCAL_MEM_FENCE)', \
                           '__kernel', '',
                           '__global', '__local', str(self.ls), \
                           self.activate_float64, '', self.code_dtype]

        for src_code_file in os.listdir(self.src_path):
            src_code = file('%s/%s' % (self.src_path, src_code_file), 'r').read()
            self.templates[src_code_file] = src_code
        # Program of controlling fields in this FDTD_world instances

#        code_prev = ['__DECLARE_CONSTANT_ARRAYS__', \
#                     'strts[i]', 'steps[i]', 'd_dev[i]', 'd_buf[i]']
        code_prev = ['__DECLARE_SHARED_ARRAYS__', \
                     'strts[i]', 'steps[i]', 'd_dev[i]', 'd_buf[i]']
        code_post = ['', \
                     'strts[i]', 'steps[i]', 'd_dev[i]', 'd_buf[i]']

        code = template_to_code(self.templates['get_set_data'], code_prev+self.code_prev, code_post+self.code_post)
        self.programs['get_set_data']  = self.cl.Program(self.ctx, code).build()

        code = template_to_code(fdtd.engine.templates['incident_direct'], fdtd.engine.code_prev, fdtd.engine.code_post)
        self.programs['src'] = self.build(code)

        code = template_to_code(fdtd.engine.templates['incident_direct'], fdtd.engine.code_prev, fdtd.engine.code_post)
        self.programs['src'] = self.build(code)
        self.kernels['src_a'] = self.programs['src'].update_from_array
        self.kernels['src_v'] = self.programs['src'].update_from_value
        self.kernels['src_m'] = self.programs['src'].update_mnch

        code = template_to_code(fdtd.engine.templates['running_fourier_transform'], fdtd.engine.code_prev, fdtd.engine.code_post)
        self.programs['rft'] = self.build(code)
        self.kernels['rft'] = self.programs['rft'].update
