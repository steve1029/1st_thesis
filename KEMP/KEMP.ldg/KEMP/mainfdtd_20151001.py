# -*- coding:utf-8 -*-
import numpy as np
from scipy.constants import c as c0

from util   import *
from ndarray import *
from datetime import datetime as dtm
engines = ('gnu_cpu', 'intel_cpu', 'intel_cpu_without_openmp', 'nvidia_cuda', 'nvidia_cuda_single_device', 'nvidia_opencl')

# This CODE includes mainfdtd classes, "Basic_FDTD"
class FDTD_single_device:
    def __init__(self):
        self.cores = {'main':self, 'dispersive':None, 'pml':None, 'pbc':None, 'src':None, 'tfsf':None}
        self.boundary_condition = {'x':{'+':None, '-':None}, 'y':{'+':None, '-':None} ,'z':{'+':None, '-':None}}
        self.ndarray_use_buffer_mode = False
        self.fdtd_group = [self]
        self.comm_mode = False
        self.is_ces = False
        self.is_chs = False

        if not self.is_subfdtd:
            self.master = True
            self.first  = True
            self.last   = True
            if self.from_external_engine:
                self.set_computing_engine(self.engines[0])
            else:
                self.set_computing_engine(self.engine_name)
            self.set_fields_single()
            self.init_single()
            self.setup()
        else:
            self.master = False
            self.first  = False
            self.last   = False

    def set_computing_engine(self, arg_engine):
        import engine as eng
        if self.from_external_engine:
            check_type('engine', arg_engine, eng.engine)
            if  'cpu' in arg_engine.name:
                if   arg_engine.name == 'intel_cpu':
                    self.engine = eng.intel_cpu()
                    self.engine.init(self)
                    self.engines = [self.engine]
                elif arg_engine.name == 'intel_cpu_without_openmp':
                    self.engine = eng.intel_cpu_without_openmp()
                    self.engine.init(self)
                    self.engines = [self.engine]
                else:
                    raise NotImplementedError, ''
            else:
                self.engine = eng.subengine(arg_engine)
        else:
            check_type('engine', arg_engine, str)
            if   arg_engine == 'nvidia_opencl':
                self.engine = eng.nvidia_opencl()
            elif arg_engine == 'nvidia_cuda':
                self.engine = eng.nvidia_cuda()
            elif arg_engine == 'gnu_cpu':
                self.engine = eng.gnu_cpu()
            elif arg_engine == 'intel_cpu':
                self.engine = eng.intel_cpu()
            elif arg_engine == 'intel_cpu_without_openmp':
                self.engine = eng.intel_cpu_without_openmp()
            else:
                import exception
                raise exception.ComputingEngineError, 'Invalid Computing Engine NAME'
            self.engine.init(self)
            self.engines = [self.engine]

    def get_info_single(self):
        info = {}
        info['MODE'] = self.mode
        info['SHAPE'] = self.shape
        info['SIZE']  = self.n

        info['Uniform_Grid'] = self.is_uniform_grid
        info['CE Array'] = self.is_ces
        info['CH Array'] = self.is_chs
        info['Dispersive'] = self.is_dispersive

        info['PML'] = self.is_pml
        if self.is_pml:
            info['PML_apply'] = self.pml.pml_apply
            info['PML_thick'] = self.pml.pml_thick
            info['PML_PARAMETER'] = {'alpha':self.pml.alpha, \
                                     'kappa':self.pml.kappa, \
                                     'alpha exponent': self.pml.alpha_exponent, \
                                     'sigma exponent': self.pml.sigma_exponent}
        info['PBC'] = self.is_pbc
        if self.is_pbc:
            info['PBC_apply'] = self.pbc.pbc_apply

        info['Computing Architecture'] = self.engine.name
        info['Computing Device'] = self.engine.device
        info['Memory size of Computing Device (MiB)'] = self.engine.max_mem_size / (1024.**2)
        info['Allocated Memory size (MiB)'] = self.engine.allocated_mem_size / (1024.**2)
        for key in sorted(info.iterkeys()):
            print '%s: %s' % (key, info[key])
        return info

    def setup(self):
        code_rds_prev = [', __GLOBAL__ __FLOAT__* rdx_e', ', __GLOBAL__ __FLOAT__* rdy_e', ', __GLOBAL__ __FLOAT__* rdz_e', \
                         ', __GLOBAL__ __FLOAT__* rdx_h', ', __GLOBAL__ __FLOAT__* rdy_h', ', __GLOBAL__ __FLOAT__* rdz_h', \
                         'rdx_e[i]*', 'rdy_e[j]*', 'rdz_e[k]*', \
                         'rdx_h[i]*', 'rdy_h[j]*', 'rdz_h[k]*']
        if self.is_uniform_grid:
            code_rds_post = ['', '', '', \
                             '', '', '', \
                             '', '', '', \
                             '', '', '']
        else:
            code_rds_post = code_rds_prev

        code_ces_prev = [', __GLOBAL__ __FLOAT__* ce1x', ', __GLOBAL__ __FLOAT__* ce1y', ', __GLOBAL__ __FLOAT__* ce1z', \
                         ', __GLOBAL__ __FLOAT__* ce2x', ', __GLOBAL__ __FLOAT__* ce2y', ', __GLOBAL__ __FLOAT__* ce2z', \
                         'ce1x[idx]*', 'ce1y[idx]*', 'ce1z[idx]*', \
                         'ce2x[idx]*', 'ce2y[idx]*', 'ce2z[idx]*']
        if not self.is_electric:
            ce2 = '%s*' % self.dt
            code_ces_post = ['', '', '', \
                             '', '', '', \
                             '', '', '', \
                             ce2, ce2, ce2]
        else:
            code_ces_post = code_ces_prev

        code_chs_prev = [', __GLOBAL__ __FLOAT__* ch1x', ', __GLOBAL__ __FLOAT__* ch1y', ', __GLOBAL__ __FLOAT__* ch1z', \
                         ', __GLOBAL__ __FLOAT__* ch2x', ', __GLOBAL__ __FLOAT__* ch2y', ', __GLOBAL__ __FLOAT__* ch2z', \
                         'ch1x[idx]*', 'ch1y[idx]*', 'ch1z[idx]*', \
                         'ch2x[idx]*', 'ch2y[idx]*', 'ch2z[idx]*']
        if not self.is_magnetic:
            ch2 = '%s*' % self.dt
            code_chs_post = ['', '', '', \
                             '', '', '', \
                             '', '', '', \
                             ch2, ch2, ch2]
        else:
            code_chs_post = code_chs_prev

        # For OpenMP guided schedule setting
        if 'cpu' in self.engine.name:
            code_rds_omp = [', rdx_e', ', rdy_e', ', rdz_e', ', rdx_h', ', rdy_h', ', rdz_h']
            code_rds_prev += code_rds_omp
            if self.is_uniform_grid: code_rds_post += ['', '', '', '', '', '']
            else: code_rds_post += code_rds_omp

            code_ces_omp = [', ce1x', ', ce1y', ', ce1z', ', ce2x', ', ce2y', ', ce2z']
            code_ces_prev += code_ces_omp
            if not self.is_electric: code_ces_post += ['', '', '', '', '', '']
            else: code_ces_post += code_ces_omp

            code_chs_omp = [', ch1x', ', ch1y', ', ch1z', ', ch2x', ', ch2y', ', ch2z']
            code_chs_prev += code_chs_omp
            if not self.is_magnetic: code_chs_post += ['', '', '', '', '', '']
            else: code_chs_post += code_chs_omp

        # Program build
        code = template_to_code(self.engine.templates['fdtd'], \
                                code_ces_prev + code_chs_prev + code_rds_prev + self.engine.code_prev, \
                                code_ces_post + code_chs_post + code_rds_post + self.engine.code_post)

        self.engine.programs['fdtd'] = self.engine.build(code)
        self.engine.updates['main_e'] = self.update_e_single
        self.engine.updates['main_h'] = self.update_h_single
        self.engine.updates['src_e']  = self.update_e_src
        self.engine.updates['src_h']  = self.update_h_src
        self.engine.updates['rft_e']  = self.update_e_rft
        self.engine.updates['rft_h']  = self.update_h_rft
        self.engine.kernel_args['main_e'] = {}
        self.engine.kernel_args['main_h'] = {}
        if self.mode == '2DTE':
            for part in self.complex_parts:
                self.engine.kernel_args['main_e'][part] = [self.engine.queue, (self.engine.gs,), (self.engine.ls,), \
                                                           np.int32(self.nx), np.int32(self.ny), \
                                                           self.ex.__dict__[part].data, \
                                                           self.ey.__dict__[part].data, \
                                                           self.hz.__dict__[part].data]
                self.engine.kernel_args['main_h'][part] = [self.engine.queue, (self.engine.gs,), (self.engine.ls,), \
                                                           np.int32(self.nx), np.int32(self.ny), \
                                                           self.ex.__dict__[part].data, \
                                                           self.ey.__dict__[part].data, \
                                                           self.hz.__dict__[part].data]
                if self.is_electric:
                    self.engine.kernel_args['main_e'][part] += [self.ce1x.data, self.ce1y.data, \
                                                                self.ce2x.data, self.ce2y.data]
                if self.is_magnetic:
                    self.engine.kernel_args['main_h'][part] += [self.ch1z.data, \
                                                                self.ch2z.data]
                if not self.is_uniform_grid:
                    self.engine.kernel_args['main_e'][part] += [self.rdx_e.data, self.rdy_e.data]
                    self.engine.kernel_args['main_h'][part] += [self.rdx_h.data, self.rdy_h.data]
            if   'opencl' in self.engine.name:
                self.engine.kernels['main_e'] = self.engine.programs['fdtd'].update_e_2dte
                self.engine.kernels['main_h'] = self.engine.programs['fdtd'].update_h_2dte
            elif   'cuda' in self.engine.name:
                self.engine.kernels['main_e'] = self.engine.get_function(self.engine.programs['fdtd'], 'update_e_2dte')
                self.engine.kernels['main_h'] = self.engine.get_function(self.engine.programs['fdtd'], 'update_h_2dte')
                self.engine.prepare(self.engine.kernels['main_e'], self.engine.kernel_args['main_e']['real'])
                self.engine.prepare(self.engine.kernels['main_h'], self.engine.kernel_args['main_h']['real'])
            elif    'cpu' in self.engine.name:
                self.engine.kernels['main_e'] = self.engine.set_kernel(self.engine.programs['fdtd'].update_e_2dte, self.engine.kernel_args['main_e']['real'])
                self.engine.kernels['main_h'] = self.engine.set_kernel(self.engine.programs['fdtd'].update_h_2dte, self.engine.kernel_args['main_h']['real'])

        elif self.mode == '2DTM':
            for part in self.complex_parts:
                self.engine.kernel_args['main_e'][part] = [self.engine.queue, (self.engine.gs,), (self.engine.ls,), \
                                                           np.int32(self.nx), np.int32(self.ny), \
                                                           self.ez.__dict__[part].data, \
                                                           self.hx.__dict__[part].data, \
                                                           self.hy.__dict__[part].data]
                self.engine.kernel_args['main_h'][part] = [self.engine.queue, (self.engine.gs,), (self.engine.ls,), \
                                                           np.int32(self.nx), np.int32(self.ny), \
                                                           self.ez.__dict__[part].data, \
                                                           self.hx.__dict__[part].data, \
                                                           self.hy.__dict__[part].data]
                if self.is_electric:
                    self.engine.kernel_args['main_e'][part] += [self.ce1z.data, \
                                                                self.ce2z.data]
                if self.is_magnetic:
                    self.engine.kernel_args['main_h'][part] += [self.ch1x.data, self.ch1y.data, \
                                                                self.ch2x.data, self.ch2y.data]
                if not self.is_uniform_grid:
                    self.engine.kernel_args['main_e'][part] += [self.rdx_e.data, self.rdy_e.data]
                    self.engine.kernel_args['main_h'][part] += [self.rdx_h.data, self.rdy_h.data]
            if   'opencl' in self.engine.name:
                self.engine.kernels['main_e'] = self.engine.programs['fdtd'].update_e_2dtm
                self.engine.kernels['main_h'] = self.engine.programs['fdtd'].update_h_2dtm
            elif   'cuda' in self.engine.name:
                self.engine.kernels['main_e'] = self.engine.get_function(self.engine.programs['fdtd'], 'update_e_2dtm')
                self.engine.kernels['main_h'] = self.engine.get_function(self.engine.programs['fdtd'], 'update_h_2dtm')
                self.engine.prepare(self.engine.kernels['main_e'], self.engine.kernel_args['main_e']['real'])
                self.engine.prepare(self.engine.kernels['main_h'], self.engine.kernel_args['main_h']['real'])
            elif    'cpu' in self.engine.name:
                self.engine.kernels['main_e'] = self.engine.set_kernel(self.engine.programs['fdtd'].update_e_2dtm, self.engine.kernel_args['main_e']['real'])
                self.engine.kernels['main_h'] = self.engine.set_kernel(self.engine.programs['fdtd'].update_h_2dtm, self.engine.kernel_args['main_h']['real'])

        elif self.mode == '3D':
            for part in self.complex_parts:
                self.engine.kernel_args['main_e'][part] = [self.engine.queue, (self.engine.gs,), (self.engine.ls,), \
                                                           np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                           self.ex.__dict__[part].data, \
                                                           self.ey.__dict__[part].data, \
                                                           self.ez.__dict__[part].data, \
                                                           self.hx.__dict__[part].data, \
                                                           self.hy.__dict__[part].data, \
                                                           self.hz.__dict__[part].data]
                self.engine.kernel_args['main_h'][part] = [self.engine.queue, (self.engine.gs,), (self.engine.ls,), \
                                                           np.int32(self.nx), np.int32(self.ny), np.int32(self.nz), \
                                                           self.ex.__dict__[part].data, \
                                                           self.ey.__dict__[part].data, \
                                                           self.ez.__dict__[part].data, \
                                                           self.hx.__dict__[part].data, \
                                                           self.hy.__dict__[part].data, \
                                                           self.hz.__dict__[part].data]
                if self.is_electric:
                    self.engine.kernel_args['main_e'][part] += [self.ce1x.data, self.ce1y.data, self.ce1z.data, \
                                                                self.ce2x.data, self.ce2y.data, self.ce2z.data]
                if self.is_magnetic:
                    self.engine.kernel_args['main_h'][part] += [self.ch1x.data, self.ch1y.data, self.ch1z.data, \
                                                                self.ch2x.data, self.ch2y.data, self.ch2z.data]
                if not self.is_uniform_grid:
                    self.engine.kernel_args['main_e'][part] += [self.rdx_e.data, self.rdy_e.data, self.rdz_e.data]
                    self.engine.kernel_args['main_h'][part] += [self.rdx_h.data, self.rdy_h.data, self.rdz_h.data]
            if   'opencl' in self.engine.name:
                self.engine.kernels['main_e'] = self.engine.programs['fdtd'].update_e_3d
                self.engine.kernels['main_h'] = self.engine.programs['fdtd'].update_h_3d
            elif   'cuda' in self.engine.name:
                self.engine.kernels['main_e'] = self.engine.get_function(self.engine.programs['fdtd'], 'update_e_3d')
                self.engine.kernels['main_h'] = self.engine.get_function(self.engine.programs['fdtd'], 'update_h_3d')
                self.engine.prepare(self.engine.kernels['main_e'], self.engine.kernel_args['main_e']['real'])
                self.engine.prepare(self.engine.kernels['main_h'], self.engine.kernel_args['main_h']['real'])
            elif    'cpu' in self.engine.name:
                self.engine.kernels['main_e'] = self.engine.set_kernel(self.engine.programs['fdtd'].update_e_3d, self.engine.kernel_args['main_e']['real'])
                self.engine.kernels['main_h'] = self.engine.set_kernel(self.engine.programs['fdtd'].update_h_3d, self.engine.kernel_args['main_h']['real'])

        for name, core in self.cores.items():
            if name != 'main' and core is not None: core.setup()

    def set_fields_single(self, x_offset=0., min_ds=None):
        real_dtype = comp_to_real(self.dtype)
        self.engine.mem_alloc_size = 0
        if '2D' in self.mode:
            # setting fields
            try:
                for eh in self.ehs:
                    eh.release_data()
            except AttributeError:
                pass
            if self.mode == '2DTE':
                names = ['ex', 'ey', 'hz']
                self.ehs = self.ex, self.ey, self.hz = \
                    [Fields(self, (self.nx, self.ny), dtype=self.dtype, name=names[i]) for i in xrange(3)]

            if self.mode == '2DTM':
                names = ['hx', 'hy', 'ez']
                self.ehs = self.hx, self.hy, self.ez = \
                    [Fields(self, (self.nx, self.ny), dtype=self.dtype, name=names[i]) for i in xrange(3)]

            self.x_SI_pts, self.y_SI_pts = \
                [np.array([0.] + list(np.array(self.space_grid[i]).cumsum())) for i in xrange(2)]
            self.x_SI_cel, self.y_SI_cel = \
                [np.zeros(self.space_grid[i].size+1, dtype=real_dtype) for i in xrange(2)]
            self.x_SI_cel[:-1] = (self.x_SI_pts[:-1] + self.x_SI_pts[1:])*.5
            self.y_SI_cel[:-1] = (self.y_SI_pts[:-1] + self.y_SI_pts[1:])*.5
            self.x_SI_cel[-1]  =  self.x_SI_cel[-2]  + self.space_grid[0][-1]
            self.y_SI_cel[-1]  =  self.y_SI_cel[-2]  + self.space_grid[1][-1]

            self.x = self.x_SI_cel
            self.y = self.y_SI_cel

            if self.mode == '2DTE':
                self.ex.x = self.x_SI_cel; self.ex.y = self.y_SI_pts;
                self.ey.x = self.x_SI_pts; self.ey.y = self.y_SI_cel;
                self.hz.x = self.x_SI_cel; self.hz.y = self.y_SI_cel;

            if self.mode == '2DTM':
                self.hx.x = self.x_SI_pts; self.hx.y = self.y_SI_cel;
                self.hy.x = self.x_SI_cel; self.hy.y = self.y_SI_pts;
                self.ez.x = self.x_SI_pts; self.ez.y = self.y_SI_pts;

            # reciprocal grids settings
            self.min_ds = np.array([self.space_grid[i].min() for i in xrange(2)]).min()
            self.delta_t = self.dt*self.min_ds/c0
            rdx_pts, rdy_pts = \
                [np.zeros(self.space_grid[i].size+1, dtype=real_dtype) for i in xrange(2)]
            rdx_pts[1:], rdy_pts[1:] = \
                [1./real_dtype(self.space_grid[i]/self.min_ds) for i in xrange(2)]
            rdx_cel, rdy_cel = \
                [np.zeros(self.space_grid[i].size+1, dtype=real_dtype) for i in xrange(2)]
            rdx_cel[:-2], rdy_cel[:-2] = \
                [2./real_dtype((self.space_grid[i][1:]+self.space_grid[i][:-1])/self.min_ds) for i in xrange(2)]
            rdx_cel[-2], rdy_cel[-2] = \
                [1./real_dtype(self.space_grid[i][-1]/self.min_ds) for i in xrange(2)]

            # coordinate settings ( for structure setting )
            self.x_cel_NU = Fields(self, (self.nx,), dtype=real_dtype, mem_flag='r', init_value=self.x_SI_cel/self.min_ds, name='x_cel_NU')
            self.y_cel_NU = Fields(self, (self.ny,), dtype=real_dtype, mem_flag='r', init_value=self.y_SI_cel/self.min_ds, name='y_cel_NU')
            self.x_pts_NU = Fields(self, (self.nx,), dtype=real_dtype, mem_flag='r', init_value=self.x_SI_pts/self.min_ds, name='x_pts_NU')
            self.y_pts_NU = Fields(self, (self.ny,), dtype=real_dtype, mem_flag='r', init_value=self.y_SI_pts/self.min_ds, name='y_pts_NU')

            self.rdx_e = Fields(self, (self.nx,), dtype=real_dtype, mem_flag='r', init_value=rdx_pts, name='rdx_e')
            self.rdy_e = Fields(self, (self.ny,), dtype=real_dtype, mem_flag='r', init_value=rdy_pts, name='rdy_e')
            self.rdx_h = Fields(self, (self.nx,), dtype=real_dtype, mem_flag='r', init_value=rdx_cel, name='rdx_h')
            self.rdy_h = Fields(self, (self.ny,), dtype=real_dtype, mem_flag='r', init_value=rdy_cel, name='rdy_h')

        elif '3D' in self.mode:
            # setting fields
            try:
                for eh in self.ehs:
                    eh.release_data()
            except AttributeError:
                pass
            self.ehs = self.ex, self.ey, self.ez, self.hx, self.hy, self.hz = \
            self.EHs = self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz = \
                [Fields(self, (self.nx, self.ny, self.nz), dtype=self.dtype, mem_flag='rw', \
                        name=['ex', 'ey', 'ez', 'hx', 'hy', 'hz'][i]) for i in xrange(6)]

#            set_coordinate(self, x_offset)

            # reciprocal grids settings
            if min_ds is not None:
                self.min_ds = min_ds
            else:
                self.min_ds = np.array([self.space_grid[i].min() for i in xrange(3)]).min()
            self.delta_t = self.dt*self.min_ds/c0
            rdx_pts, rdy_pts, rdz_pts = \
                [np.ones(self.space_grid[i].size+1, dtype=real_dtype) for i in xrange(3)]
            rdx_pts[:-1], rdy_pts[:-1], rdz_pts[:-1] = \
                [1./real_dtype(self.space_grid[i]/self.min_ds) for i in xrange(3)]
            rdx_cel, rdy_cel, rdz_cel = \
                [np.ones(self.space_grid[i].size+1, dtype=real_dtype) for i in xrange(3)]
            rdx_cel[1:-1], rdy_cel[1:-1], rdz_cel[1:-1] = \
                [2./real_dtype((self.space_grid[i][1:]+self.space_grid[i][:-1])/self.min_ds) for i in xrange(3)]
            rdx_cel[-1], rdy_cel[-1], rdz_cel[-1] = \
                [1./real_dtype(self.space_grid[i][-1]/self.min_ds) for i in xrange(3)]

            self.x_SI_pts, self.y_SI_pts, self.z_SI_pts = \
                [np.array([0.] + list(np.array(self.space_grid[i]).cumsum())) for i in xrange(3)]
            self.x_SI_cel, self.y_SI_cel, self.z_SI_cel = \
                [np.zeros(self.space_grid[i].size+1, dtype=real_dtype) for i in xrange(3)]
            self.x_SI_cel[:-1] = (self.x_SI_pts[:-1] + self.x_SI_pts[1:])*.5
            self.y_SI_cel[:-1] = (self.y_SI_pts[:-1] + self.y_SI_pts[1:])*.5
            self.z_SI_cel[:-1] = (self.z_SI_pts[:-1] + self.z_SI_pts[1:])*.5
            self.x_SI_cel[-1]  =  self.x_SI_cel[-2]  + self.space_grid[0][-1]
            self.y_SI_cel[-1]  =  self.y_SI_cel[-2]  + self.space_grid[1][-1]
            self.z_SI_cel[-1]  =  self.z_SI_cel[-2]  + self.space_grid[2][-1]
            self.x_SI_pts += x_offset
            self.x_SI_cel += x_offset

            self.x = self.x_SI_cel
            self.y = self.y_SI_cel
            self.z = self.z_SI_cel
            # coordinate settings ( for structure setting )
            self.x_cel_NU = Fields(self, (self.nx,), dtype=real_dtype, mem_flag='r', init_value=self.x_SI_cel/self.min_ds, name='x_cel_NU')
            self.y_cel_NU = Fields(self, (self.ny,), dtype=real_dtype, mem_flag='r', init_value=self.y_SI_cel/self.min_ds, name='y_cel_NU')
            self.z_cel_NU = Fields(self, (self.nz,), dtype=real_dtype, mem_flag='r', init_value=self.z_SI_cel/self.min_ds, name='z_cel_NU')
            self.x_pts_NU = Fields(self, (self.nx,), dtype=real_dtype, mem_flag='r', init_value=self.x_SI_pts/self.min_ds, name='x_pts_NU')
            self.y_pts_NU = Fields(self, (self.ny,), dtype=real_dtype, mem_flag='r', init_value=self.y_SI_pts/self.min_ds, name='y_pts_NU')
            self.z_pts_NU = Fields(self, (self.nz,), dtype=real_dtype, mem_flag='r', init_value=self.z_SI_pts/self.min_ds, name='z_pts_NU')

            self.rdx_e = Fields(self, (self.nx,), dtype=real_dtype, mem_flag='r', init_value=rdx_cel, name='rdx_e')
            self.rdy_e = Fields(self, (self.ny,), dtype=real_dtype, mem_flag='r', init_value=rdy_cel, name='rdy_e')
            self.rdz_e = Fields(self, (self.nz,), dtype=real_dtype, mem_flag='r', init_value=rdz_cel, name='rdz_e')
            self.rdx_h = Fields(self, (self.nx,), dtype=real_dtype, mem_flag='r', init_value=rdx_pts, name='rdx_h')
            self.rdy_h = Fields(self, (self.ny,), dtype=real_dtype, mem_flag='r', init_value=rdy_pts, name='rdy_h')
            self.rdz_h = Fields(self, (self.nz,), dtype=real_dtype, mem_flag='r', init_value=rdz_pts, name='rdz_h')

    def init_single(self, opt='fields'):
        for rft in self.rfts:
            rft.init_single()
        for source in self.sources:
            source.init_single()
        if 'fields' in opt:
            if self.mode == '2DTE':
                self.ex[:,:] = 0.
                self.ey[:,:] = 0.
                self.hz[:,:] = 0.
                if self.cores['pml'] is not None:
                    if '-' in self.cores['pml'].pml_apply['x']:
                        self.cores['pml'].psi_eyx_m[:,:] = 0.
                        self.cores['pml'].psi_hzx_m[:,:] = 0.
                    if '+' in self.cores['pml'].pml_apply['x']:
                        self.cores['pml'].psi_eyx_p[:,:] = 0.
                        self.cores['pml'].psi_hzx_p[:,:] = 0.
                    if '-' in self.cores['pml'].pml_apply['y']:
                        self.cores['pml'].psi_exy_m[:,:] = 0.
                        self.cores['pml'].psi_hzy_m[:,:] = 0.
                    if '+' in self.cores['pml'].pml_apply['y']:
                        self.cores['pml'].psi_exy_p[:,:] = 0.
                        self.cores['pml'].psi_hzy_p[:,:] = 0.
                if self.cores['dispersive'] is not None:
                    for fx in self.cores['dispersive'].dr_fx_r:
                        fx[:,:] = 0.
                    for fy in self.cores['dispersive'].dr_fy_r:
                        fy[:,:] = 0.
                    for fx in self.cores['dispersive'].cp_fx_r:
                        fx[:,:] = 0.
                    for fy in self.cores['dispersive'].cp_fy_r:
                        fy[:,:] = 0.
                    for fx in self.cores['dispersive'].cp_fx_i:
                        fx[:,:] = 0.
                    for fy in self.cores['dispersive'].cp_fy_i:
                        fy[:,:] = 0.

            if self.mode == '2DTM':
                self.hx[:,:] = 0.
                self.hy[:,:] = 0.
                self.ez[:,:] = 0.
                if self.cores['pml'] is not None:
                    if '-' in self.cores['pml'].pml_apply['x']:
                        self.cores['pml'].psi_ezx_m[:,:] = 0.
                        self.cores['pml'].psi_hyx_m[:,:] = 0.
                    if '+' in self.cores['pml'].pml_apply['x']:
                        self.cores['pml'].psi_ezx_p[:,:] = 0.
                        self.cores['pml'].psi_hyx_p[:,:] = 0.
                    if '-' in self.cores['pml'].pml_apply['y']:
                        self.cores['pml'].psi_ezy_m[:,:] = 0.
                        self.cores['pml'].psi_hxy_m[:,:] = 0.
                    if '+' in self.cores['pml'].pml_apply['y']:
                        self.cores['pml'].psi_ezy_p[:,:] = 0.
                        self.cores['pml'].psi_hxy_p[:,:] = 0.
                if self.cores['dispersive'] is not None:
                    for fz in self.cores['dispersive'].dr_fz_r:
                        fz[(0,0),(-1,-1)] = 0.
                    for fz in self.cores['dispersive'].cp_fz_r:
                        fz[(0,0),(-1,-1)] = 0.
                    for fz in self.cores['dispersive'].cp_fz_i:
                        fz[(0,0),(-1,-1)] = 0.

            elif self.mode == '3D':
                self.ex[:,:,:] = 0.
                self.ey[:,:,:] = 0.
                self.ez[:,:,:] = 0.
                self.hx[:,:,:] = 0.
                self.hy[:,:,:] = 0.
                self.hz[:,:,:] = 0.
                if self.cores['pml'] is not None:
                    if '-' in self.cores['pml'].pml_apply['x']:
                        self.cores['pml'].psi_eyx_m[:,:,:] = 0.
                        self.cores['pml'].psi_ezx_m[:,:,:] = 0.
                        self.cores['pml'].psi_hyx_m[:,:,:] = 0.
                        self.cores['pml'].psi_hzx_m[:,:,:] = 0.
                    if '+' in self.cores['pml'].pml_apply['x']:
                        self.cores['pml'].psi_eyx_p[:,:,:] = 0.
                        self.cores['pml'].psi_ezx_p[:,:,:] = 0.
                        self.cores['pml'].psi_hyx_p[:,:,:] = 0.
                        self.cores['pml'].psi_hzx_p[:,:,:] = 0.
                    if '-' in self.cores['pml'].pml_apply['y']:
                        self.cores['pml'].psi_ezy_m[:,:,:] = 0.
                        self.cores['pml'].psi_exy_m[:,:,:] = 0.
                        self.cores['pml'].psi_hzy_m[:,:,:] = 0.
                        self.cores['pml'].psi_hxy_m[:,:,:] = 0.
                    if '+' in self.cores['pml'].pml_apply['y']:
                        self.cores['pml'].psi_ezy_p[:,:,:] = 0.
                        self.cores['pml'].psi_exy_p[:,:,:] = 0.
                        self.cores['pml'].psi_hzy_p[:,:,:] = 0.
                        self.cores['pml'].psi_hxy_p[:,:,:] = 0.
                    if '-' in self.cores['pml'].pml_apply['z']:
                        self.cores['pml'].psi_exz_m[:,:,:] = 0.
                        self.cores['pml'].psi_eyz_m[:,:,:] = 0.
                        self.cores['pml'].psi_hxz_m[:,:,:] = 0.
                        self.cores['pml'].psi_hyz_m[:,:,:] = 0.
                    if '+' in self.cores['pml'].pml_apply['z']:
                        self.cores['pml'].psi_exz_p[:,:,:] = 0.
                        self.cores['pml'].psi_eyz_p[:,:,:] = 0.
                        self.cores['pml'].psi_hxz_p[:,:,:] = 0.
                        self.cores['pml'].psi_hyz_p[:,:,:] = 0.
                if self.cores['dispersive'] is not None:
                    for fx in self.cpwd.dr_fx_r:
                        fx[:,:,:] = 0.
                    for fy in self.cpwd.dr_fy_r:
                        fy[:,:,:] = 0.
                    for fz in self.cpwd.dr_fz_r:
                        fz[:,:,:] = 0.
                    for fx in self.cpwd.cp_fx_r:
                        fx[:,:,:] = 0.
                    for fy in self.cpwd.cp_fy_r:
                        fy[:,:,:] = 0.
                    for fz in self.cpwd.cp_fz_r:
                        fz[:,:,:] = 0.
                    for fx in self.cpwd.cp_fx_i:
                        fx[:,:,:] = 0.
                    for fy in self.cpwd.cp_fy_i:
                        fy[:,:,:] = 0.
                    for fz in self.cpwd.cp_fz_i:
                        fz[:,:,:] = 0.

        if 'structures' in opt:
            if self.is_electric:
                if   self.mode == '2DTE':
                    self.ce1x[:,:] = 1.
                    self.ce1y[:,:] = 1.
                    self.ce2x[:,:] = .5
                    self.ce2y[:,:] = .5
                elif self.mode == '2DTM':
                    self.ce1z[:,:] = 1.
                    self.ce2z[:,:] = .5
                elif self.mode == '3D':
                    self.ce1x[:,:,:] = 1.
                    self.ce1y[:,:,:] = 1.
                    self.ce1z[:,:,:] = 1.
                    self.ce2x[:,:,:] = .5
                    self.ce2y[:,:,:] = .5
                    self.ce2z[:,:,:] = .5
            if self.is_magnetic:
                if   self.mode == '2DTE':
                    self.ch1z[:,:] = 1.
                    self.ch2z[:,:] = .5
                elif self.mode == '2DTM':
                    self.ch1x[:,:] = 1.
                    self.ch1y[:,:] = 1.
                    self.ch2x[:,:] = .5
                    self.ch2y[:,:] = .5
                elif self.mode == '3D':
                    self.ch1x[:,:,:] = 1.
                    self.ch1y[:,:,:] = 1.
                    self.ch1z[:,:,:] = 1.
                    self.ch2x[:,:,:] = .5
                    self.ch2y[:,:,:] = .5
                    self.ch2z[:,:,:] = .5

    def apply_PML_single(self, pml_apply, pml_thick, alpha0, kappa0, malpha, msigma, lnR0):
        from pml import PML
        self.pml = PML(self, pml_apply, pml_thick, alpha0, kappa0, malpha, msigma, lnR0)

    def apply_PBC_single(self, pbc_apply, klx=0., kly=0., klz=0.):
        from pbc import PBC, BBC
        if klx**2 + kly**2 + klz**2 > 0. and not self.is_complex:
            raise exception.BoundaryConditionError, 'To use BBC, set complex FDTD'
        if not self.is_complex:
            self.pbc = PBC(self, pbc_apply)
        else:
            self.pbc = BBC(self, pbc_apply, klx=klx, kly=kly, klz=klz)

    def apply_TFSF_single(self, region, boundary={'x':'+-', 'y':'+-', 'z':'+-'}, is_oblique=True):
        from incident import TFSF_Boundary
        tfsf = TFSF_Boundary(self, region, boundary, is_oblique)
        self.tfsfs.append(tfsf)
        self.is_tfsf = True
        return tfsf.tfsf_fdtd

    def apply_TFSF1D_single(self, field, region, rot_vec, pol_vec, boundary={'x':'+-', 'y':'+-', 'z':'+-'}, material=None):
        from incident import TFSF_1D_Boundary
        tfsf = TFSF_1D_Boundary(self, field, region, rot_vec, pol_vec, boundary, material)
        self.tfsfs.append(tfsf)
        self.is_tfsf = True
        return tfsf.tfsf_inc

    def apply_direct_source_single(self, field, region):
        from incident import DirectSource
        return DirectSource(self, field, region)

    def apply_monochromatic_source_single(self, field, region, freq):
        from incident import MonochromaticSource
        return MonochromaticSource(self, field, region, freq)

    def apply_RFT_single(self, field, region, freq_domain):
        from rft import RFT
        return RFT(self, field, region, freq_domain)

    def set_structures_single(self, structures=None, wait=True):
        real_dtype = comp_to_real(self.dtype)
        self.dp_materials = []
        if structures is not None:
            for structure in structures:
                if structure.material.classification in ['dielectric', 'dielectromagnetic', 'electric dispersive', 'electromagnetic dispersive']:
                    self.is_electric = True
                if structure.material.classification in ['dimagnetic', 'dielectromagnetic', 'magnetic dispersive', 'electromagnetic dispersive']:
                    self.is_magnetic = True
                if structure.material.classification in ['electric dispersive', 'magnetic dispersive', 'electromagnetic dispersive']:
                    self.is_dispersive = True

                if   self.is_electric and self.is_magnetic and self.is_dispersive:
                    self.materials_classification = 'electromagnetic_dispersive'
                elif self.is_electric and self.is_magnetic:
                    self.materials_classification = 'dielectromagnetic'
                elif self.is_electric and self.is_dispersive:
                    self.materials_classification = 'electric dispersive'
                elif self.is_magnetic and self.is_dispersive:
                    self.materials_classification = 'magnetic dispersive'
                elif self.is_electric:
                    self.materials_classification = 'dielectric'
                elif self.is_magnetic:
                    self.materials_classification = 'dimagnetic'

                if   structure.material.classification in ['dielectric', 'dimagnetic', 'dielectromagnetic']:
                    structure.material.set_coeff(self)
                elif 'dispersive' in structure.material.classification:
                    if self.dp_materials.count(structure.material) == 0:
                        self.dp_materials.append(structure.material)

            if self.is_electric:
                if not self.is_ces:
                    self.ce1s = self.ce1x, self.ce1y, self.ce1z = \
                    self.cE1s = self.cE1x, self.cE1y, self.cE1z = \
                        [Fields(self, self.shape, dtype=real_dtype, mem_flag='rw', \
                                init_value=np.ones(self.shape, dtype=real_dtype), \
                                name=['ce1x', 'ce1y', 'ce1z'][i]) for i in xrange(3)]
                    self.ce2s = self.ce2x, self.ce2y, self.ce2z = \
                    self.cE2s = self.cE2x, self.cE2y, self.cE2z = \
                        [Fields(self, self.shape, dtype=real_dtype, mem_flag='rw', \
                                init_value=np.ones(self.shape, dtype=real_dtype)*self.dt, \
                                name=['ce2x', 'ce2y', 'ce2z'][i]) for i in xrange(3)]
                    self.ces  = self.cEs = self.ce1s + self.ce2s
                    self.is_ces = True
            if self.is_magnetic:
                if not self.is_chs:
                    self.ch1s = self.ch1x, self.ch1y, self.ch1z = \
                    self.cH1s = self.cH1x, self.cH1y, self.cH1z = \
                        [Fields(self, self.shape, dtype=real_dtype, mem_flag='rw', \
                                init_value=np.ones(self.shape, dtype=real_dtype), \
                                name=['ch1x', 'ch1y', 'ch1z'][i]) for i in xrange(3)]
                    self.ch2s = self.ch2x, self.ch2y, self.ch2z = \
                    self.cH2s = self.cH2x, self.cH2y, self.cH2z = \
                        [Fields(self, self.shape, dtype=real_dtype, mem_flag='rw', \
                                init_value=np.ones(self.shape, dtype=real_dtype)*self.dt, \
                                name=['ch2x', 'ch2y', 'ch2z'][i]) for i in xrange(3)]
                    self.chs = self.cHs = self.ch1s + self.ch2s
                    self.is_chs = True

            if self.is_dispersive:
                import dispersive as dp
                for material in self.dp_materials:
                    for pole in material.poles:
                        pole.set_coeff(self)
                    material.set_coeff(self)

                '''
                if   self.mode == '2DTE':
                    self.ex.release_data()
                    self.ey.release_data()
                    self.hz.release_data()
                elif self.mode == '2DTM':
                    self.hx.release_data()
                    self.hy.release_data()
                    self.ez.release_data()
                elif self.mode == '3D':
                    self.ex.release_data()
                    self.ey.release_data()
                    self.ez.release_data()
                    self.hx.release_data()
                    self.hy.release_data()
                    self.hz.release_data()
                '''
                self.mkex, self.mkey, self.mkez = \
                    [Fields(self, self.shape, dtype=np.int32, mem_flag='rw', \
                            init_value=np.zeros(self.shape, dtype=np.int32), \
                            name='mark') for i in xrange(3)]

            for structure in structures:
                structure.set_structure(self)

            if self.is_dispersive:
                if self.cores['dispersive'] is not None:
                    for fx in self.cpwd.dr_fx_r: fx.release_data()
                    for fy in self.cpwd.dr_fy_r: fy.release_data()
                    for fz in self.cpwd.dr_fz_r: fz.release_data()
                    for fx in self.cpwd.cp_fx_r: fx.release_data()
                    for fy in self.cpwd.cp_fy_r: fy.release_data()
                    for fz in self.cpwd.cp_fz_r: fz.release_data()
                    for fx in self.cpwd.cp_fx_i: fx.release_data()
                    for fy in self.cpwd.cp_fy_i: fy.release_data()
                    for fz in self.cpwd.cp_fz_i: fz.release_data()
                region_true = False
                mx, my, mz = [np.zeros(self.shape, dtype=np.int32) for i in xrange(3)]

                if 'opencl' in self.engine.name:
                    self.engine.cl.enqueue_read_buffer(self.engine.queue, self.mkex.data, mx).wait()
                    self.engine.cl.enqueue_read_buffer(self.engine.queue, self.mkey.data, my).wait()
                    self.engine.cl.enqueue_read_buffer(self.engine.queue, self.mkez.data, mz).wait()
                elif 'cuda' in self.engine.name:
                    self.engine.enqueue(self.engine.drv.memcpy_dtoh, [mx, self.mkex.data]).wait()
                    self.engine.enqueue(self.engine.drv.memcpy_dtoh, [my, self.mkey.data]).wait()
                    self.engine.enqueue(self.engine.drv.memcpy_dtoh, [mz, self.mkez.data]).wait()
                elif  'cpu' in self.engine.name:
                    mx = self.mkex.data 
                    my = self.mkey.data 
                    mz = self.mkez.data 
                if   '2D' in self.mode:
                    pt0 = np.array([self.nx, self.ny])
                    pt1 = np.zeros(2, dtype=int)
                    mx_x, mx_y = [mx.sum(1-i) for i in xrange(2)]
                    my_x, my_y = [my.sum(1-i) for i in xrange(2)]
                    mxyx = mx_x + my_x
                    mxyy = mx_y + my_y
                    for i in xrange(self.nx):
                        if mxyx[i] > 0:
                            region_true = True
                            if i < pt0[0]: pt0[0] = i
                            if i > pt1[0]: pt1[0] = i
                    for j in xrange(self.ny):
                        if mxyy[j] > 0:
                            region_true = True
                            if j < pt0[1]: pt0[1] = j
                            if j > pt1[1]: pt1[1] = j
#                    for i in xrange(2):
#                        if pt0[i] == 0: pt0[i] = 1
                    if region_true:
                        self.cpwd = dp.Drude_CP_world(self, (tuple(pt0), tuple(pt1)), self.dp_materials)
                        cpwd = self.cpwd
                        args = [self.engine.queue, (cpwd.gs,), (self.engine.ls,), \
                                np.int32(self.nx), np.int32(self.ny), \
                                np.int32(cpwd.nx), np.int32(cpwd.ny), \
                                np.int32(cpwd.px), np.int32(cpwd.py), \
                                self.mkex.data, self.mkey.data, self.mkez.data, \
                                cpwd.mx.data, cpwd.my.data, cpwd.mz.data]
                        if 'opencl' in self.engine.name:
                            evt = structures[-1].prg.mark_to_mk_2d(*args)
                        elif 'cuda' in self.engine.name:
                            func = self.engine.get_function(structures[-1].prg, 'mark_to_mk_2d')
                            self.engine.prepare(func, args)
                            evt = self.engine.enqueue_kernel(func, args)
                        elif  'cpu' in self.engine.name:
                            self.engine.set_kernel(structures[-1].prg.mark_to_mk_2d, args)
                            evt = structures[-1].prg.mark_to_mk_2d(*(args[3:]))
                        evt.wait()
                elif '3D' in self.mode:
                    pt0 = np.array([self.nx, self.ny, self.nz])
                    pt1 = np.zeros(3, dtype=int)
                    ms = [mx, my, mz]
                    mx_x, my_x, mz_x = [ms[i].sum(1).sum(1) for i in xrange(3)]
                    mx_y, my_y, mz_y = [ms[i].sum(2).sum(0) for i in xrange(3)]
                    mx_z, my_z, mz_z = [ms[i].sum(0).sum(0) for i in xrange(3)]
                    mxyzx = mx_x + my_x + mz_x
                    mxyzy = mx_y + my_y + mz_y
                    mxyzz = mx_z + my_z + mz_z
                    for i in xrange(self.nx):
                        if mxyzx[i] > 0:
                            region_true = True
                            if i < pt0[0]: pt0[0] = i
                            if i > pt1[0]: pt1[0] = i
                    for j in xrange(self.ny):
                        if mxyzy[j] > 0:
                            region_true = True
                            if j < pt0[1]: pt0[1] = j
                            if j > pt1[1]: pt1[1] = j
                    for k in xrange(self.nz):
                        if mxyzz[k] > 0:
                            region_true = True
                            if k < pt0[2]: pt0[2] = k
                            if k > pt1[2]: pt1[2] = k
#                    for i in xrange(3):
#                        if pt0[i] == 0: pt0[i] = 1
                    if region_true:
                        self.cpwd = dp.Drude_CP_world(self, (tuple(pt0), tuple(pt1)), self.dp_materials)
                        cpwd = self.cpwd
                        args = [self.engine.queue, (cpwd.gs,), (self.engine.ls,), \
                                np.int32(self.nx), np.int32(self.ny), np.int32(self.nz),
                                np.int32(cpwd.nx), np.int32(cpwd.ny), np.int32(cpwd.nz),
                                np.int32(cpwd.px), np.int32(cpwd.py), np.int32(cpwd.pz),
                                self.mkex.data, self.mkey.data, self.mkez.data,
                                cpwd.mx.data, cpwd.my.data, cpwd.mz.data]
                        if 'opencl' in self.engine.name:
                            evt = structures[-1].prg.mark_to_mk_3d(*args)
                        elif 'cuda' in self.engine.name:
                            func = self.engine.get_function(structures[-1].prg, 'mark_to_mk_3d')
                            self.engine.prepare(func, args)
                            evt = self.engine.enqueue_kernel(func, args)
                        elif  'cpu' in self.engine.name:
                            self.engine.set_kernel(structures[-1].prg.mark_to_mk_3d, args)
                            structures[-1].prg.mark_to_mk_3d(*(args[3:]))
                            evt = FakeEvent()
                        evt.wait()
                '''
                self.mkex.release_data()
                self.mkey.release_data()
                self.mkez.release_data()
                fd_buf = np.zeros(self.shape, dtype=comp_to_real(self.dtype))
                if   self.mode == '2DTE':
                    self.ex.set_buffer_from_FDTD(fd_buf, self)
                    self.ey.set_buffer_from_FDTD(fd_buf, self)
                    self.hz.set_buffer_from_FDTD(fd_buf, self)
                elif self.mode == '2DTM':
                    self.hx.set_buffer_from_FDTD(fd_buf, self)
                    self.hy.set_buffer_from_FDTD(fd_buf, self)
                    self.ez.set_buffer_from_FDTD(fd_buf, self)
                elif self.mode == '3D':
                    self.ex.set_buffer_from_FDTD(fd_buf, self)
                    self.ey.set_buffer_from_FDTD(fd_buf, self)
                    self.ez.set_buffer_from_FDTD(fd_buf, self)
                    self.hx.set_buffer_from_FDTD(fd_buf, self)
                    self.hy.set_buffer_from_FDTD(fd_buf, self)
                    self.hz.set_buffer_from_FDTD(fd_buf, self)
                '''

            '''
            if self.is_ces:
                if   '2D' in self.mode:
                    self.ce1x[:,0] = 1.
                    self.ce1y[0,:] = 1.
                    self.ce1z[0,:] = 1.
                    self.ce1z[:,0] = 1.
                    self.ce2x[:,0] = self.dt
                    self.ce2y[0,:] = self.dt
                    self.ce2z[0,:] = self.dt
                    self.ce2z[:,0] = self.dt
                elif '3D' in self.mode:
                    self.ce1x[:,0,:] = 1.
                    self.ce1x[:,:,0] = 1.
                    self.ce1y[:,:,0] = 1.
                    self.ce1y[0,:,:] = 1.
                    self.ce1z[0,:,:] = 1.
                    self.ce1z[:,0,:] = 1.
                    self.ce1x[:,0,:] = self.dt
                    self.ce1x[:,:,0] = self.dt
                    self.ce1y[:,:,0] = self.dt
                    self.ce1y[0,:,:] = self.dt
                    self.ce1z[0,:,:] = self.dt
                    self.ce1z[:,0,:] = self.dt
                evts = []
                if 'opencl' in self.engine.name:
                    evts.append(self.engine.cl.enqueue_read_buffer(self.engine.queue, self.ce1x.data, self.ce1x.buff))
                    evts.append(self.engine.cl.enqueue_read_buffer(self.engine.queue, self.ce1y.data, self.ce1y.buff))
                    evts.append(self.engine.cl.enqueue_read_buffer(self.engine.queue, self.ce1z.data, self.ce1z.buff))
                    evts.append(self.engine.cl.enqueue_read_buffer(self.engine.queue, self.ce2x.data, self.ce2x.buff))
                    evts.append(self.engine.cl.enqueue_read_buffer(self.engine.queue, self.ce2y.data, self.ce2y.buff))
                    evts.append(self.engine.cl.enqueue_read_buffer(self.engine.queue, self.ce2z.data, self.ce2z.buff))
                elif 'cuda' in self.engine.name:
                    evts.append(self.engine.enqueue(self.engine.drv.memcpy_dtoh, [self.ce1x.buff, self.ce1x.data]))
                    evts.append(self.engine.enqueue(self.engine.drv.memcpy_dtoh, [self.ce1y.buff, self.ce1y.data]))
                    evts.append(self.engine.enqueue(self.engine.drv.memcpy_dtoh, [self.ce1z.buff, self.ce1z.data]))
                    evts.append(self.engine.enqueue(self.engine.drv.memcpy_dtoh, [self.ce2x.buff, self.ce2x.data]))
                    evts.append(self.engine.enqueue(self.engine.drv.memcpy_dtoh, [self.ce2y.buff, self.ce2y.data]))
                    evts.append(self.engine.enqueue(self.engine.drv.memcpy_dtoh, [self.ce2z.buff, self.ce2z.data]))
                wait_for_events(self, evts)
            if self.is_chs:
                if   '2D' in self.mode:
                    self.ch1x[:,-1] = self.dt
                    self.ch1y[-1,:] = self.dt
                    self.ch1z[-1,:] = self.dt
                    self.ch1z[:,-1] = self.dt
                    self.ch2x[:,-1] = 0.
                    self.ch2y[-1,:] = 0.
                    self.ch2z[-1,:] = 0.
                    self.ch2z[:,-1] = 0.
                elif '3D' in self.mode:
                    self.ch1x[:,-1,:] = self.dt
                    self.ch1x[:,:,-1] = self.dt
                    self.ch1y[:,:,-1] = self.dt
                    self.ch1y[-1,:,:] = self.dt
                    self.ch1z[-1,:,:] = self.dt
                    self.ch1z[:,-1,:] = self.dt
                    self.ch1x[:,-1,:] = 0.
                    self.ch1x[:,:,-1] = 0.
                    self.ch1y[:,:,-1] = 0.
                    self.ch1y[-1,:,:] = 0.
                    self.ch1z[-1,:,:] = 0.
                    self.ch1z[:,-1,:] = 0.
                evts = []
                if 'opencl' in self.engine.name:
                    evts.append(self.engine.cl.enqueue_read_buffer(self.engine.queue, self.ch1x.data, self.ch1x.buff))
                    evts.append(self.engine.cl.enqueue_read_buffer(self.engine.queue, self.ch1y.data, self.ch1y.buff))
                    evts.append(self.engine.cl.enqueue_read_buffer(self.engine.queue, self.ch1z.data, self.ch1z.buff))
                    evts.append(self.engine.cl.enqueue_read_buffer(self.engine.queue, self.ch2x.data, self.ch2x.buff))
                    evts.append(self.engine.cl.enqueue_read_buffer(self.engine.queue, self.ch2y.data, self.ch2y.buff))
                    evts.append(self.engine.cl.enqueue_read_buffer(self.engine.queue, self.ch2z.data, self.ch2z.buff))
                elif 'cuda' in self.engine.name:
                    evts.append(self.engine.enqueue(self.engine.drv.memcpy_dtoh, [self.ch1x.buff, self.ch1x.data]))
                    evts.append(self.engine.enqueue(self.engine.drv.memcpy_dtoh, [self.ch1y.buff, self.ch1y.data]))
                    evts.append(self.engine.enqueue(self.engine.drv.memcpy_dtoh, [self.ch1z.buff, self.ch1z.data]))
                    evts.append(self.engine.enqueue(self.engine.drv.memcpy_dtoh, [self.ch2x.buff, self.ch2x.data]))
                    evts.append(self.engine.enqueue(self.engine.drv.memcpy_dtoh, [self.ch2y.buff, self.ch2y.data]))
                    evts.append(self.engine.enqueue(self.engine.drv.memcpy_dtoh, [self.ch2z.buff, self.ch2z.data]))
                wait_for_events(self, evts)
            '''
            self.setup()

    def update_e_single(self, wait=True):
        evts = []
        for part in self.complex_parts:
            if   'opencl' in self.engine.name:
                evt = self.engine.kernels['main_e'](*(self.engine.kernel_args['main_e'][part]))
            elif   'cuda' in self.engine.name:
                evt = self.engine.enqueue_kernel(self.engine.kernels['main_e'], self.engine.kernel_args['main_e'][part], False)
            elif    'cpu' in self.engine.name:
                evt = self.engine.kernels['main_e'](*(self.engine.kernel_args['main_e'][part][3:]))
            evts.append(evt)
        if wait:
            wait_for_events(self, evts)
        return evts

    def update_h_single(self, wait=True):
        evts = []
        for part in self.complex_parts:
            if   'opencl' in self.engine.name:
                evt = self.engine.kernels['main_h'](*(self.engine.kernel_args['main_h'][part]))
            elif   'cuda' in self.engine.name:
                t0 = dtm.now()
                evt = self.engine.enqueue_kernel(self.engine.kernels['main_h'], self.engine.kernel_args['main_h'][part], False)
            elif    'cpu' in self.engine.name:
                evt = self.engine.kernels['main_h'](*(self.engine.kernel_args['main_h'][part][3:]))
            evts.append(evt)
        if wait:
            wait_for_events(self, evts)
        return evts

    def sync_fields(self, wait=True):
        evts = []
        for field in self.fields_list:
            if field.buff_updated_flag:
                evts += field.buff_to_data(wait=wait)
                field.buff_updated_flag = False
                if field.is_complex:
                    field.real.buff_updated_flag = False
                    field.imag.buff_updated_flag = False
        if wait: wait_for_events(self, evts)
        return evts

    def update_e_flags(self):
        if   self.mode == '2DTE':
            self.ex.data_updated_flag = True
            self.ey.data_updated_flag = True
            if self.is_complex:
                self.ex.real.data_updated_flag = True
                self.ex.imag.data_updated_flag = True
                self.ey.real.data_updated_flag = True
                self.ey.imag.data_updated_flag = True
        elif self.mode == '2DTM':
            self.ez.data_updated_flag = True
            if self.is_complex:
                self.ez.real.data_updated_flag = True
                self.ez.imag.data_updated_flag = True
        elif self.mode == '3D':
            self.ex.data_updated_flag = True
            self.ey.data_updated_flag = True
            self.ez.data_updated_flag = True
            if self.is_complex:
                self.ex.real.data_updated_flag = True
                self.ex.imag.data_updated_flag = True
                self.ey.real.data_updated_flag = True
                self.ey.imag.data_updated_flag = True
                self.ez.real.data_updated_flag = True
                self.ez.imag.data_updated_flag = True

    def update_h_flags(self):
        if   self.mode == '2DTE':
            self.hz.data_updated_flag = True
            if self.is_complex:
                self.hz.real.data_updated_flag = True
                self.hz.imag.data_updated_flag = True
        elif self.mode == '2DTM':
            self.hx.data_updated_flag = True
            self.hy.data_updated_flag = True
            if self.is_complex:
                self.hx.real.data_updated_flag = True
                self.hx.imag.data_updated_flag = True
                self.hy.real.data_updated_flag = True
                self.hy.imag.data_updated_flag = True
        elif self.mode == '3D':
            self.hx.data_updated_flag = True
            self.hy.data_updated_flag = True
            self.hz.data_updated_flag = True
            if self.is_complex:
                self.hx.real.data_updated_flag = True
                self.hx.imag.data_updated_flag = True
                self.hy.real.data_updated_flag = True
                self.hy.imag.data_updated_flag = True
                self.hz.real.data_updated_flag = True
                self.hz.imag.data_updated_flag = True

    def updateE_single(self, wait=True):
        t0 = dtm.now()
        evts = []
        if not self.is_subfdtd:
            self.sync_fields()
        t1 = dtm.now()
        if self.is_tfsf and not self.is_subfdtd:
            for tfsf in self.tfsfs:
                evts += tfsf.tfsf_fdtd.updateE(wait=wait)
        for priority in self.engine.priority_order_e:
            if self.engine.updates[priority] != None:
                evts += self.engine.updates[priority](wait=False)
        if wait: wait_for_events(self, evts)
        t2 = dtm.now()
        self.update_e_flags()
        self.updateE_calc_time =  t2 - t1
        self.updateE_exch_time =  t1 - t0
        self.updateE_comm_time =        0
        self.updateE_time      =  t2 - t0
        return evts

    def updateH_single(self, wait=True):
        t0 = dtm.now()
        evts = []
        if not self.is_subfdtd:
            self.sync_fields()
        t1 = dtm.now()
        if self.is_tfsf and not self.is_subfdtd:
            for tfsf in self.tfsfs:
                evts += tfsf.tfsf_fdtd.updateH(wait=wait)
        for priority in self.engine.priority_order_h:
            if self.engine.updates[priority] != None:
                evts += self.engine.updates[priority](wait=False)
        if wait: wait_for_events(self, evts)
        t2 = dtm.now()
        self.update_h_flags()
        self.updateH_calc_time =  t2 - t1
        self.updateH_exch_time =  t1 - t0
        self.updateH_comm_time =        0
        self.updateH_time      =  t2 - t0
        return evts

    def update_e_src(self, wait=True):
        evts = []
        for source in self.sources:
            evtlist = source.update_e(wait=wait)
            evts += evtlist
        return evts

    def update_h_src(self, wait=True):
        evts = []
        for source in self.sources:
            evtlist = source.update_h(wait=wait)
            evts += evtlist
        return evts

    def update_e_rft(self, wait=True):
        evts = []
        for rft in self.rfts:
            evtlist = rft.update_e(wait=wait)
            evts += evtlist
        return evts

    def update_h_rft(self, wait=True):
        evts = []
        for rft in self.rfts:
            evtlist = rft.update_h(wait=wait)
            evts += evtlist
        return evts

class FDTD_multi_devices:
    def __init__(self):
        self.cores = {'main':self, 'pole':None, 'dispersive':None, 'pml':None, 'pbc':None, 'tfsf':None}
        self.pbc_x = False
        self.comm_mode = False
        self.is_ces = False
        self.is_chs = False
        if not self.is_subfdtd:
            self.master = True
            self.first  = True
            self.last   = True

        self.fdtd_group  = []
        self.nx_group, self.x_strt_group, self.x_stop_group = divide_xgrid(self.space_grid[0], self.num_devices)

        if not self.from_external_engine:
            self.engines = []
        for i in xrange(self.num_devices):
            sub_space_grid = (self.space_grid[0][self.x_strt_group[i]:self.x_stop_group[i]], \
                self.space_grid[1], self.space_grid[2])
            if self.from_external_engine is True:
                sub_fdtd = Basic_FDTD_3d(self.mode, sub_space_grid, dtype=self.dtype, engine=[self.engines[i]], device_id=0, is_subfdtd=True)
            else:
                sub_fdtd = Basic_FDTD_3d(self.mode, sub_space_grid, dtype=self.dtype, engine=self.engine_name, device_id=self.device_id[i], is_subfdtd=True)
            self.fdtd_group.append(sub_fdtd)

        self.x_SI_pts, self.y_SI_pts, self.z_SI_pts = \
            [np.array([0.] + list(np.array(self.space_grid[i]).cumsum())) for i in xrange(3)]

        if not self.is_subfdtd:
            self.set_fields_multi()

    def get_info_multi(self):
        info = {}
        info['MODE'] = self.mode
        info['SHAPE'] = self.shape
        info['SIZE']  = self.n
        info['Distribution of x-axis grid'] = \
            [(self.x_strt_group[i], self.x_stop_group[i]) for i in xrange(self.num_devices)]

        info['Uniform_Grid'] = self.is_uniform_grid
        info['CE Array'] = self.is_ces
        info['CH Array'] = self.is_chs
        info['Dispersive'] = self.is_dispersive

        info['PML'] = self.is_pml
        if self.is_pml:
            info['PML_apply'] = self.pml_apply
            info['PML_thick'] = self.pml_thick
            info['PML_PARAMETER'] = {'alpha':self.alpha, \
                                     'kappa':self.kappa, \
                                     'alpha_exponent': self.alpha_exponent, \
                                     'sigma_exponent': self.sigma_exponent}
        info['PBC'] = self.is_pbc
        if self.is_pbc:
            info['PBC_apply'] = self.pbc_apply

        sinfos = []
        for sfdtd in self.fdtd_group:
            sinfos.append(sfdtd.get_info_single())

        for key in sorted(info.iterkeys()):
            print '%s: %s' % (key, info[key])

        for num, sinfo in enumerate(sinfos):
            print 'Assigned x grid(Device %s): (%s to %s)' % \
                  (num, self.x_strt_group[num], self.x_stop_group[num])

        items = ['SHAPE', \
                 'SIZE', \
                 'Computing Architecture', \
                 'Computing Device', \
                 'Memory size of Computing Device (MiB)', \
                 'Allocated Memory size (MiB)']
        for item in items:
            for num, sinfo in enumerate(sinfos):
                print '%s(Device %s): %s' % (item, num, sinfo[item])

        return info, sinfos

    def init_multi(self, opt='fields'):
        for i in xrange(self.num_devices):
            self.fdtd_group[i].init_single(opt)

    def set_fields_multi(self, x_offset=0., min_ds=None):
        real_dtype = comp_to_real(self.dtype)
        if min_ds is not None:
            self.min_ds = min_ds
        else:
            self.min_ds = np.array([self.space_grid[i].min() for i in xrange(3)]).min()
        self.delta_t = self.dt*self.min_ds/c0
        if self.from_external_engine:
            for i in xrange(self.num_devices):
                sub_fdtd = self.fdtd_group[i]
                sub_fdtd.set_computing_engine(self.engines[i])
        else:
            self.engines = []
            for i in xrange(self.num_devices):
                sub_fdtd = self.fdtd_group[i]
                sub_fdtd.set_computing_engine(self.engine_name)
                self.engines.append(sub_fdtd.engine)
        for i in xrange(self.num_devices):
            sub_fdtd = self.fdtd_group[i]
            sub_fdtd.set_fields_single(x_offset=x_offset+self.x_SI_pts[self.x_strt_group[i]], min_ds=self.min_ds)
            sub_fdtd.init_single()
            sub_fdtd.setup()
            sub_fdtd.master = False
            if i == 0:
                sub_fdtd.first = True
            if i == self.num_devices - 1:
                sub_fdtd.last  = True

        name = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']
        self.ehs = self.ex, self.ey, self.ez, self.hx, self.hy, self.hz = \
        self.EHs = self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz = \
            [Fields_multi_devices(self, (self.nx, self.ny, self.nz), self.dtype, name=name[i]) for i in xrange(6)]

        set_coordinate(self, x_offset)
        self.init_exchange_multi()

    def apply_PML_multi(self, pml_apply, pml_thick, alpha0, kappa0, malpha, msigma, lnR0):
        from pml import PML
        self.pml_apply = pml_apply
        self.pml_thick = pml_thick
        self.alpha = alpha0
        self.kappa = kappa0
        self.alpha_exponent = malpha
        self.sigma_exponent = msigma
        for i in xrange(self.num_devices):
            sub_pml_apply = {'x':'', 'y':pml_apply['y'], 'z':pml_apply['z']}
            if   i == 0:
                if '-' in pml_apply['x']:
                    sub_pml_apply['x'] += '-'
            if i == self.num_devices - 1:
                if '+' in pml_apply['x']:
                    sub_pml_apply['x'] += '+'
            subfdtd = self.fdtd_group[i]
            subfdtd.apply_PML_single(sub_pml_apply, pml_thick, alpha0, kappa0, malpha, msigma, lnR0)

    def apply_PBC_multi(self, pbc_apply, klx=0., kly=0., klz=0.):
        from pbc import PBC, BBC
        self.pbc_apply = pbc_apply
        self.pbc_klx = klx
        self.sin_klx = np.sin(klx)
        self.cos_klx = np.cos(klx)
        if pbc_apply['x']:
            self.pbc_x   = True
            for sfdtd in self.fdtd_group:
                for part in self.complex_parts:
                    sfdtd.engine.kernel_args['buf_to_field']['ey'][part] = [sfdtd.engine.queue, (gs,), (sfdtd.engine.ls,), \
                                                                            np.int32(0), np.int32(0*2+0), np.int32(1*2+0), \
                                                                            np.int32(sfdtd.nx), np.int32(sfdtd.ny), np.int32(sfdtd.nz), \
                                                                            comp_to_real(fdtd.dtype)(-self.sin_klx), \
                                                                            comp_to_real(fdtd.dtype)(+self.cos_klx), \
                                                                            sfdtd.ey.__dict__[part].data, sfdtd.buff_recv_e]
                    sfdtd.engine.kernel_args['buf_to_field']['ez'][part] = [sfdtd.engine.queue, (gs,), (sfdtd.engine.ls,), \
                                                                            np.int32(0), np.int32(0*2+1), np.int32(1*2+1), \
                                                                            np.int32(sfdtd.nx), np.int32(sfdtd.ny), np.int32(sfdtd.nz), \
                                                                            comp_to_real(fdtd.dtype)(-self.sin_klx), \
                                                                            comp_to_real(fdtd.dtype)(+self.cos_klx), \
                                                                            sfdtd.ez.__dict__[part].data, sfdtd.buff_recv_e]
                    sfdtd.engine.kernel_args['buf_to_field']['hy'][part] = [sfdtd.engine.queue, (gs,), (sfdtd.engine.ls,), \
                                                                            np.int32(1), np.int32(0*2+0), np.int32(1*2+0), \
                                                                            np.int32(sfdtd.nx), np.int32(sfdtd.ny), np.int32(sfdtd.nz), \
                                                                            comp_to_real(fdtd.dtype)(+self.sin_klx), \
                                                                            comp_to_real(fdtd.dtype)(+self.cos_klx), \
                                                                            sfdtd.hy.__dict__[part].data, sfdtd.buff_recv_h]
                    sfdtd.engine.kernel_args['buf_to_field']['hz'][part] = [sfdtd.engine.queue, (gs,), (sfdtd.engine.ls,), \
                                                                            np.int32(1), np.int32(0*2+1), np.int32(1*2+1), \
                                                                            np.int32(sfdtd.nx), np.int32(sfdtd.ny), np.int32(sfdtd.nz), \
                                                                            comp_to_real(fdtd.dtype)(+self.sin_klx), \
                                                                            comp_to_real(fdtd.dtype)(+self.cos_klx), \
                                                                            sfdtd.hz.__dict__[part].data, sfdtd.buff_recv_h]

            if 'opencl' in sfdtd.engine.name:
                sfdtd.engine.kernels['buf_to_field'] = sfdtd.engine.programs['get_set_data'].buf_to_field_bbc
            elif 'cuda' in sfdtd.engine.name:
                sfdtd.engine.kernels['buf_to_field'] = sfdtd.engine.get_function(sfdtd.engine.programs['get_set_data'], 'buf_to_field_bbc')
                sfdtd.engine.prepare(sfdtd.engine.kernels['buf_to_field_bbc'], sfdtd.engine.kernel_args['buf_to_field_bbc']['ey']['real'])

        else:
            self.pbc_x   = False
        for i in xrange(self.num_devices):
            sub_pbc_apply = {'x':False, 'y':pbc_apply['y'], 'z':pbc_apply['z']}
            subfdtd = self.fdtd_group[i]
            subfdtd.apply_PBC(sub_pbc_apply, klx=klx, kly=kly, klz=klz)

    def apply_TFSF_multi(self, region, boundary={'x':'+-', 'y':'+-', 'z':'+-'}, is_oblique=True):
        from incident import TFSF_Boundary
        tfsf = TFSF_Boundary(self, region, boundary, is_oblique)
        self.tfsfs.append(tfsf)
        self.is_tfsf = True
        return tfsf.tfsf_fdtd

    def apply_TFSF1D_multi(self, field, region, rot_vec, pol_vec, boundary={'x':'+-', 'y':'+-', 'z':'+-'}, material=None):
        from incident import TFSF_1D_Boundary
        tfsf = TFSF_1D_Boundary(self, field, region, rot_vec, pol_vec, boundary, material)
        self.tfsfs.append(tfsf)
        self.is_tfsf = True
        return tfsf.tfsf_inc

    def apply_direct_source_multi(self, field, region):
        from incident import DirectSource
        return DirectSource(self, field, region)

    def apply_monochromatic_source_multi(self, field, region, freq):
        from incident import MonochromaticSource
        return MonochromaticSource(self, field, region, freq)

    def apply_RFT_multi(self, field, region, freq_domain):
        from rft import RFT
        return RFT(self, field, region, freq_domain)

    def set_structures_multi(self, structures=None):
        if structures is not None:
            for i in xrange(self.num_devices):
                subfdtd = self.fdtd_group[i]
                subfdtd.set_structures_single(structures)
        for structure in structures:
            if structure.material.classification in ['dielectric', 'electric_dispersive', 'electromagnetic_dispersive']:
                self.is_electric = True
            if structure.material.classification in ['dimagnetic', 'magnetic_dispersive', 'electromagnetic_dispersive']:
                self.is_magnetic = True
            if structure.material.classification in ['electric_dispersive', 'magnetic_dispersive', 'electromagnetic_dispersive']:
                self.is_dispersive = True
        real_dtype = comp_to_real(self.dtype)
        if self.is_electric:
            name = ['ce1x', 'ce1y', 'ce1z', 'ce2x', 'ce2y', 'ce2z']
            self.ces = self.ce1x, self.ce1y, self.ce1z, self.ce2x, self.ce2y, self.ce2z = \
            self.cEs = self.cE1x, self.cE1y, self.cE1z, self.cE1x, self.cE2y, self.cE2z = \
                [Fields_multi_devices(self, self.shape, real_dtype, name=name[i]) for i in xrange(6)]
            self.ce1s = self.ces[:3]
            self.cE1s = self.ces[:3]
            self.ce2s = self.ces[3:]
            self.cE2s = self.ces[3:]
            self.is_ces = True
        if self.is_magnetic:
            name = ['ch1x', 'ch1y', 'ch1z', 'ch2x', 'ch2y', 'ch2z']
            self.chs = self.ch1x, self.ch1y, self.ch1z, self.ch2x, self.ch2y, self.ch2z = \
            self.cHs = self.cH1x, self.cH1y, self.cH1z, self.cH2x, self.cH2y, self.cH2z = \
                [Fields_multi_devices(self, self.shape, real_dtype, name=name[i]) for i in xrange(6)]
            self.ch1s = self.chs[:3]
            self.cH1s = self.chs[:3]
            self.ch2s = self.chs[3:]
            self.cH2s = self.chs[3:]
            self.is_chs = True

    def update_e_multi(self, wait=True):
        evts = []
        for i in xrange(self.num_devices):
            evts += self.fdtd_group[i].update_e(wait=False)
        if wait: wait_for_events(self, evts)
        return evts

    def update_h_multi(self, wait=True):
        evts = []
        for i in xrange(self.num_devices):
            evts += self.fdtd_group[i].update_h(wait=False)
        if wait: wait_for_events(self, evts)
        return evts

    def init_exchange_multi(self):
        real_dtype = comp_to_real(self.dtype)
        dsize = {np.float32:4, np.float64:8}[real_dtype]
        for sfdtd in self.fdtd_group:
            if not self.is_complex:
#                sfdtd.temp_send_e, sfdtd.temp_send_h, sfdtd.temp_recv_e, sfdtd.temp_recv_h = \
#                    [np.zeros((1,2,self.ny,self.nz), dtype=real_dtype) for i in xrange(4)]
                sfdtd.temp_send_e, sfdtd.temp_send_h = \
                    [np.zeros((1,2,self.ny,self.nz), dtype=real_dtype) for i in xrange(2)]
                if 'opencl' in sfdtd.engine.name:
                    mf = sfdtd.engine.cl.mem_flags
                    sfdtd.buff_send_e, sfdtd.buff_send_h, sfdtd.buff_recv_e, sfdtd.buff_recv_h = \
                        [sfdtd.engine.cl.Buffer(sfdtd.engine.ctx, mf.READ_WRITE, 2*self.ny*self.nz*dsize) for i in xrange(4)]
                elif 'cuda' in sfdtd.engine.name:
                    sfdtd.buff_send_e, sfdtd.buff_send_h, sfdtd.buff_recv_e, sfdtd.buff_recv_h = \
                        [sfdtd.engine.mem_alloc(                                 2*self.ny*self.nz*dsize) for i in xrange(4)]
                elif 'cpu' in sfdtd.engine.name:
                    sfdtd.buff_send_e, sfdtd.buff_send_h, sfdtd.buff_recv_e, sfdtd.buff_recv_h = \
                        sfdtd.temp_send_e, sfdtd.temp_send_h, sfdtd.temp_recv_e, sfdtd.temp_recv_h
            else:
#                sfdtd.temp_send_e, sfdtd.temp_send_h, sfdtd.temp_recv_e, sfdtd.temp_recv_h = \
#                    [np.zeros((2,2,self.ny,self.nz), dtype=real_dtype) for i in xrange(4)]
                sfdtd.temp_send_e, sfdtd.temp_send_h = \
                    [np.zeros((2,2,self.ny,self.nz), dtype=real_dtype) for i in xrange(2)]
                if 'opencl' in sfdtd.engine.name:
                    mf = sfdtd.engine.cl.mem_flags
                    sfdtd.buff_send_e, sfdtd.buff_send_h, sfdtd.buff_recv_e, sfdtd.buff_recv_h = \
                        [sfdtd.engine.cl.Buffer(sfdtd.engine.ctx, mf.READ_WRITE, 4*self.ny*self.nz*dsize) for i in xrange(4)]
                elif 'cuda' in sfdtd.engine.name:
                    sfdtd.buff_send_e, sfdtd.buff_send_h, sfdtd.buff_recv_e, sfdtd.buff_recv_h = \
                        [sfdtd.engine.mem_alloc(                                 4*self.ny*self.nz*dsize) for i in xrange(4)]
                elif 'cpu' in sfdtd.engine.name:
                    sfdtd.buff_send_e, sfdtd.buff_send_h, sfdtd.buff_recv_e, sfdtd.buff_recv_h = \
                        sfdtd.temp_send_e, sfdtd.temp_send_h, sfdtd.temp_recv_e, sfdtd.temp_recv_h


            sfdtd.engine.kernel_args['field_to_buf'] = {'ey':{'real':None, 'imag':None}, 'ez':{'real':None, 'imag':None}, \
                                                        'hy':{'real':None, 'imag':None}, 'hz':{'real':None, 'imag':None}  }
            sfdtd.engine.kernel_args['buf_to_field'] = {'ey':{'real':None, 'imag':None}, 'ez':{'real':None, 'imag':None}, \
                                                        'hy':{'real':None, 'imag':None}, 'hz':{'real':None, 'imag':None}  }
            sfdtd.engine.kernel_args['buf_to_field_bbc'] = {'ey':{'real':None, 'imag':None}, 'ez':{'real':None, 'imag':None}, \
                                                            'hy':{'real':None, 'imag':None}, 'hz':{'real':None, 'imag':None}  }

            if 'opencl' in sfdtd.engine.name:
                gs    = cl_global_size(self.ny*self.nz, sfdtd.engine.ls)
            else:
                gs    = self.ny*self.nz
            for i, part in enumerate(sfdtd.complex_parts):
                sfdtd.engine.kernel_args['field_to_buf']['ey'][part] = [sfdtd.engine.queue, (gs,), (sfdtd.engine.ls,), \
                                                                        np.int32(1), np.int32(i*2+0), \
                                                                        np.int32(sfdtd.nx), np.int32(sfdtd.ny), np.int32(sfdtd.nz), \
                                                                        sfdtd.ey.__dict__[part].data, sfdtd.buff_send_e]
                sfdtd.engine.kernel_args['field_to_buf']['ez'][part] = [sfdtd.engine.queue, (gs,), (sfdtd.engine.ls,), \
                                                                        np.int32(1), np.int32(i*2+1), \
                                                                        np.int32(sfdtd.nx), np.int32(sfdtd.ny), np.int32(sfdtd.nz), \
                                                                        sfdtd.ez.__dict__[part].data, sfdtd.buff_send_e]
                sfdtd.engine.kernel_args['field_to_buf']['hy'][part] = [sfdtd.engine.queue, (gs,), (sfdtd.engine.ls,), \
                                                                        np.int32(0), np.int32(i*2+0), \
                                                                        np.int32(sfdtd.nx), np.int32(sfdtd.ny), np.int32(sfdtd.nz), \
                                                                        sfdtd.hy.__dict__[part].data, sfdtd.buff_send_h]
                sfdtd.engine.kernel_args['field_to_buf']['hz'][part] = [sfdtd.engine.queue, (gs,), (sfdtd.engine.ls,), \
                                                                        np.int32(0), np.int32(i*2+1), \
                                                                        np.int32(sfdtd.nx), np.int32(sfdtd.ny), np.int32(sfdtd.nz), \
                                                                        sfdtd.hz.__dict__[part].data, sfdtd.buff_send_h]
                sfdtd.engine.kernel_args['buf_to_field']['ey'][part] = [sfdtd.engine.queue, (gs,), (sfdtd.engine.ls,), \
                                                                        np.int32(0), np.int32(i*2+0), \
                                                                        np.int32(sfdtd.nx), np.int32(sfdtd.ny), np.int32(sfdtd.nz), \
                                                                        sfdtd.ey.__dict__[part].data, sfdtd.buff_recv_e]
                sfdtd.engine.kernel_args['buf_to_field']['ez'][part] = [sfdtd.engine.queue, (gs,), (sfdtd.engine.ls,), \
                                                                        np.int32(0), np.int32(i*2+1), \
                                                                        np.int32(sfdtd.nx), np.int32(sfdtd.ny), np.int32(sfdtd.nz), \
                                                                        sfdtd.ez.__dict__[part].data, sfdtd.buff_recv_e]
                sfdtd.engine.kernel_args['buf_to_field']['hy'][part] = [sfdtd.engine.queue, (gs,), (sfdtd.engine.ls,), \
                                                                        np.int32(1), np.int32(i*2+0), \
                                                                        np.int32(sfdtd.nx), np.int32(sfdtd.ny), np.int32(sfdtd.nz), \
                                                                        sfdtd.hy.__dict__[part].data, sfdtd.buff_recv_h]
                sfdtd.engine.kernel_args['buf_to_field']['hz'][part] = [sfdtd.engine.queue, (gs,), (sfdtd.engine.ls,), \
                                                                        np.int32(1), np.int32(i*2+1), \
                                                                        np.int32(sfdtd.nx), np.int32(sfdtd.ny), np.int32(sfdtd.nz), \
                                                                        sfdtd.hz.__dict__[part].data, sfdtd.buff_recv_h]

            if 'opencl' in sfdtd.engine.name:
                sfdtd.engine.kernels['field_to_buf'] = sfdtd.engine.programs['get_set_data'].field_to_buf
                sfdtd.engine.kernels['buf_to_field'] = sfdtd.engine.programs['get_set_data'].buf_to_field
            elif 'cuda' in sfdtd.engine.name:
                sfdtd.engine.kernels['field_to_buf'] = sfdtd.engine.get_function(sfdtd.engine.programs['get_set_data'], 'field_to_buf')
                sfdtd.engine.kernels['buf_to_field'] = sfdtd.engine.get_function(sfdtd.engine.programs['get_set_data'], 'buf_to_field')
                sfdtd.engine.prepare(sfdtd.engine.kernels['field_to_buf'], sfdtd.engine.kernel_args['field_to_buf']['ey']['real'])
                sfdtd.engine.prepare(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['ey']['real'])

        for i in xrange(self.num_devices):
            if   i == self.num_devices-1:
                self.fdtd_group[  0].temp_recv_e = self.fdtd_group[i  ].temp_send_e
            else:
                self.fdtd_group[i+1].temp_recv_e = self.fdtd_group[i  ].temp_send_e

            if   i == 0:
                self.fdtd_group[ -1].temp_recv_h = self.fdtd_group[i  ].temp_send_h
            else:
                self.fdtd_group[  i].temp_recv_h = self.fdtd_group[i+1].temp_send_h


    def send_e_multi(self, wait=True):
        evts = []
        for i in xrange(self.num_devices):
            sfdtd = self.fdtd_group[i]
            for j, part in enumerate(self.complex_parts):
                if 'opencl' in sfdtd.engine.name:
                    evt = sfdtd.engine.kernels['field_to_buf'](*sfdtd.engine.kernel_args['field_to_buf']['ey'][part])
                    evts.append(evt)
                    evt = sfdtd.engine.kernels['field_to_buf'](*sfdtd.engine.kernel_args['field_to_buf']['ez'][part])
                    evts.append(evt)
                    evt = sfdtd.engine.cl.enqueue_read_buffer(sfdtd.engine.queue, sfdtd.buff_send_e, sfdtd.temp_send_e)
                    evts.append(evt)
                elif 'cuda' in sfdtd.engine.name:
                    evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['field_to_buf'], sfdtd.engine.kernel_args['field_to_buf']['ey'][part])
                    evts.append(evt)
                    evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['field_to_buf'], sfdtd.engine.kernel_args['field_to_buf']['ez'][part])
                    evts.append(evt)
                    evt = sfdtd.engine.enqueue(sfdtd.engine.drv.memcpy_dtoh_async, [sfdtd.temp_send_e, sfdtd.buff_send_e], sfdtd.engine.stream)
                    evts.append(evt)
        if wait: wait_for_events(self, evts)
        return evts

    def comm_e_multi(self):
        pass
#        if self.pbc_x:
#            if not self.is_complex:
#                self.fdtd_group[0].temp_recv_e[:,:,:,:] = self.fdtd_group[-1].temp_send_e[:,:,:,:]
#            else:
#                # real part of ey
#                self.fdtd_group[0].temp_recv_e[0,0,:,:] = + self.cos_klx*self.fdtd_group[-1].temp_send_e[0,0,:,:] \
#                                                          + self.sin_klx*self.fdtd_group[-1].temp_send_e[1,0,:,:]
#                # real part of ez
#                self.fdtd_group[0].temp_recv_e[0,1,:,:] = + self.cos_klx*self.fdtd_group[-1].temp_send_e[0,1,:,:] \
#                                                          + self.sin_klx*self.fdtd_group[-1].temp_send_e[1,1,:,:]
#                # imag part of ey
#                self.fdtd_group[0].temp_recv_e[1,0,:,:] = - self.sin_klx*self.fdtd_group[-1].temp_send_e[0,0,:,:] \
#                                                          + self.cos_klx*self.fdtd_group[-1].temp_send_e[1,0,:,:]
#                # imag part of ez
#                self.fdtd_group[0].temp_recv_e[1,1,:,:] = - self.sin_klx*self.fdtd_group[-1].temp_send_e[0,1,:,:] \
#                                                          + self.cos_klx*self.fdtd_group[-1].temp_send_e[1,1,:,:]
#
#        for i in xrange(self.num_devices-1):
#            self.fdtd_group[i+1].temp_recv_e[:] = self.fdtd_group[i].temp_send_e[:]

    def recv_e_multi(self, wait=True):
        evts = []
        for i in xrange(1, self.num_devices):
            sfdtd = self.fdtd_group[i]
            for j, part in enumerate(self.complex_parts):
                if 'opencl' in sfdtd.engine.name:
                    evt = sfdtd.engine.cl.enqueue_write_buffer(sfdtd.engine.queue, sfdtd.buff_recv_e, sfdtd.temp_recv_e)
                    evts.append(evt)
                    evt = sfdtd.engine.kernels['buf_to_field'](*sfdtd.engine.kernel_args['buf_to_field']['ey'][part])
                    evts.append(evt)
                    evt = sfdtd.engine.kernels['buf_to_field'](*sfdtd.engine.kernel_args['buf_to_field']['ez'][part])
                    evts.append(evt)
                elif 'cuda' in sfdtd.engine.name:
                    evt = sfdtd.engine.enqueue(sfdtd.engine.drv.memcpy_htod_async, [sfdtd.buff_recv_e, sfdtd.temp_recv_e], sfdtd.engine.stream)
                    evts.append(evt)
                    evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['ey'][part])
                    evts.append(evt)
                    evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['ez'][part])
                    evts.append(evt)
        if self.pbc_x:
            sfdtd = self.fdtd_group[0]
            if not self.is_complex:
                for j, part in enumerate(self.complex_parts):
                    if 'opencl' in sfdtd.engine.name:
                        evt = sfdtd.engine.cl.enqueue_write_buffer(sfdtd.engine.queue, sfdtd.buff_recv_e, sfdtd.temp_recv_e)
                        evts.append(evt)
                        evt = sfdtd.engine.kernels['buf_to_field'](*sfdtd.engine.kernel_args['buf_to_field']['ey'][part])
                        evts.append(evt)
                        evt = sfdtd.engine.kernels['buf_to_field'](*sfdtd.engine.kernel_args['buf_to_field']['ez'][part])
                        evts.append(evt)
                    elif 'cuda' in sfdtd.engine.name:
                        evt = sfdtd.engine.enqueue(sfdtd.engine.drv.memcpy_htod_async, [sfdtd.buff_recv_e, sfdtd.temp_recv_e], sfdtd.engine.stream)
                        evts.append(evt)
                        evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['ey'][part])
                        evts.append(evt)
                        evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['ez'][part])
                        evts.append(evt)
            else:
                for j, part in enumerate(self.complex_parts):
                    if 'opencl' in sfdtd.engine.name:
                        evt = sfdtd.engine.cl.enqueue_write_buffer(sfdtd.engine.queue, sfdtd.buff_recv_e, sfdtd.temp_recv_e)
                        evts.append(evt)
                        evt = sfdtd.engine.kernels['buf_to_field_bbc'](*sfdtd.engine.kernel_args['buf_to_field_bbc']['ey'][part])
                        evts.append(evt)
                        evt = sfdtd.engine.kernels['buf_to_field_bbc'](*sfdtd.engine.kernel_args['buf_to_field_bbc']['ez'][part])
                        evts.append(evt)
                    elif 'cuda' in sfdtd.engine.name:
                        evt = sfdtd.engine.enqueue(sfdtd.engine.drv.memcpy_htod_async, [sfdtd.buff_recv_e, sfdtd.temp_recv_e], sfdtd.engine.stream)
                        evts.append(evt)
                        evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field_bbc'], sfdtd.engine.kernel_args['buf_to_field_bbc']['ey'][part])
                        evts.append(evt)
                        evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field_bbc'], sfdtd.engine.kernel_args['buf_to_field_bbc']['ez'][part])
                        evts.append(evt)
        if wait: wait_for_events(self, evts)
        return evts

    def exchange_fields_e_multi(self):
        self.send_e_multi()
#        self.comm_e_multi()
        self.recv_e_multi()

    def send_h_multi(self, wait=True):
        evts = []
        for i in xrange(self.num_devices):
            sfdtd = self.fdtd_group[i]
            for j, part in enumerate(self.complex_parts):
                if 'opencl' in sfdtd.engine.name:
                    evt = sfdtd.engine.kernels['field_to_buf'](*sfdtd.engine.kernel_args['field_to_buf']['hy'][part])
                    evts.append(evt)
                    evt = sfdtd.engine.kernels['field_to_buf'](*sfdtd.engine.kernel_args['field_to_buf']['hz'][part])
                    evts.append(evt)
                    evt = sfdtd.engine.cl.enqueue_read_buffer(sfdtd.engine.queue, sfdtd.buff_send_h, sfdtd.temp_send_h)
                    evts.append(evt)
                elif 'cuda' in sfdtd.engine.name:
                    evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['field_to_buf'], sfdtd.engine.kernel_args['field_to_buf']['hy'][part])
                    evts.append(evt)
                    evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['field_to_buf'], sfdtd.engine.kernel_args['field_to_buf']['hz'][part])
                    evts.append(evt)
                    evt = sfdtd.engine.enqueue(sfdtd.engine.drv.memcpy_dtoh_async, [sfdtd.temp_send_h, sfdtd.buff_send_h], sfdtd.engine.stream)
                    evts.append(evt)
        wait_for_events(self, evts)

#    def comm_h_multi(self):
#        if self.pbc_x:
#            if not self.is_complex:
#                self.fdtd_group[-1].temp_recv_h[:,:,:,:] = self.fdtd_group[0].temp_send_h[:,:,:,:]
#            else:
#                nn = self.ny*self.nz
#                # real part of hy
#                self.fdtd_group[-1].temp_recv_h[0,0,:,:] = + self.cos_klx*self.fdtd_group[0].temp_send_h[0,0,:,:] \
#                                                           - self.sin_klx*self.fdtd_group[0].temp_send_h[1,0,:,:]
#                # real part of hz
#                self.fdtd_group[-1].temp_recv_h[0,1,:,:] = + self.cos_klx*self.fdtd_group[0].temp_send_h[0,1,:,:] \
#                                                           - self.sin_klx*self.fdtd_group[0].temp_send_h[1,1,:,:]
#                # imag part of hy
#                self.fdtd_group[-1].temp_recv_h[1,0,:,:] = + self.sin_klx*self.fdtd_group[0].temp_send_h[0,0,:,:] \
#                                                           + self.cos_klx*self.fdtd_group[0].temp_send_h[1,0,:,:]
#                # imag part of hz
#                self.fdtd_group[-1].temp_recv_h[1,1,:,:] = + self.sin_klx*self.fdtd_group[0].temp_send_h[0,1,:,:] \
#                                                           + self.cos_klx*self.fdtd_group[0].temp_send_h[1,1,:,:]
#        for i in xrange(self.num_devices-1):
#            self.fdtd_group[i].temp_recv_h[:] = self.fdtd_group[i+1].temp_send_h[:]

    def recv_h_multi(self, wait=True):
        evts = []
        for i in xrange(0, self.num_devices-1):
            sfdtd = self.fdtd_group[i]
            for j, part in enumerate(self.complex_parts):
                if 'opencl' in sfdtd.engine.name:
                    evt = sfdtd.engine.cl.enqueue_write_buffer(sfdtd.engine.queue, sfdtd.buff_recv_h, sfdtd.temp_recv_h)
                    evts.append(evt)
                    evt = sfdtd.engine.kernels['buf_to_field'](*sfdtd.engine.kernel_args['buf_to_field']['hy'][part])
                    evts.append(evt)
                    evt = sfdtd.engine.kernels['buf_to_field'](*sfdtd.engine.kernel_args['buf_to_field']['hz'][part])
                    evts.append(evt)
                elif 'cuda' in sfdtd.engine.name:
                    evt = sfdtd.engine.enqueue(sfdtd.engine.drv.memcpy_htod_async, [sfdtd.buff_recv_h, sfdtd.temp_recv_h], sfdtd.engine.stream)
                    evts.append(evt)
                    evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['hy'][part])
                    evts.append(evt)
                    evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['hz'][part])
                    evts.append(evt)
        if self.pbc_x:
            sfdtd = self.fdtd_group[-1]
            if not self.is_complex:
                for j, part in enumerate(self.complex_parts):
                    if 'opencl' in sfdtd.engine.name:
                        evt = sfdtd.engine.cl.enqueue_write_buffer(sfdtd.engine.queue, sfdtd.buff_recv_h, sfdtd.temp_recv_h)
                        evts.append(evt)
                        evt = sfdtd.engine.kernels['buf_to_field'](*sfdtd.engine.kernel_args['buf_to_field']['hy'][part])
                        evts.append(evt)
                        evt = sfdtd.engine.kernels['buf_to_field'](*sfdtd.engine.kernel_args['buf_to_field']['hz'][part])
                        evts.append(evt)
                    elif 'cuda' in sfdtd.engine.name:
                        evt = sfdtd.engine.enqueue(sfdtd.engine.drv.memcpy_htod_async, [sfdtd.buff_recv_h, sfdtd.temp_recv_h], sfdtd.engine.stream)
                        evts.append(evt)
                        evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['hy'][part])
                        evts.append(evt)
                        evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['hz'][part])
                        evts.append(evt)
            else:
                for j, part in enumerate(self.complex_parts):
                    if 'opencl' in sfdtd.engine.name:
                        evt = sfdtd.engine.cl.enqueue_write_buffer(sfdtd.engine.queue, sfdtd.buff_recv_h, sfdtd.temp_recv_h)
                        evts.append(evt)
                        evt = sfdtd.engine.kernels['buf_to_field_bbc'](*sfdtd.engine.kernel_args['buf_to_field_bbc']['hy'][part])
                        evts.append(evt)
                        evt = sfdtd.engine.kernels['buf_to_field_bbc'](*sfdtd.engine.kernel_args['buf_to_field_bbc']['hz'][part])
                        evts.append(evt)
                    elif 'cuda' in sfdtd.engine.name:
                        evt = sfdtd.engine.enqueue(sfdtd.engine.drv.memcpy_htod_async, [sfdtd.buff_recv_h, sfdtd.temp_recv_h], sfdtd.engine.stream)
                        evts.append(evt)
                        evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field_bbc'], sfdtd.engine.kernel_args['buf_to_field_bbc']['hy'][part])
                        evts.append(evt)
                        evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field_bbc'], sfdtd.engine.kernel_args['buf_to_field_bbc']['hz'][part])
                        evts.append(evt)
        if wait: wait_for_events(self, evts)
        return evts

    def exchange_fields_h_multi(self):
        self.send_h_multi()
#        self.comm_h_multi()
        self.recv_h_multi()

    def updateE_multi(self, wait=True):
        evts = []
        t0 = dtm.now()
        if not self.is_subfdtd:
            for i in xrange(self.num_devices):
                evts += self.fdtd_group[i].sync_fields(wait=False)
            wait_for_events(self, evts)
        t1 = dtm.now()
        if self.is_tfsf and not self.is_subfdtd:
            for tfsf in self.tfsfs:
                tfsf.tfsf_fdtd.updateE(wait=True)
        for priority in self.fdtd_group[0].engine.priority_order_e:
            for i in xrange(self.num_devices):
                if self.fdtd_group[i].engine.updates[priority] is not None:
                    evts += self.fdtd_group[i].engine.updates[priority](wait=False)
        if wait: wait_for_events(self, evts)
        t2 = dtm.now()
        if not self.is_subfdtd:
            self.exchange_fields_e_multi()
            for i in xrange(self.num_devices):
                self.fdtd_group[i].update_e_flags()
        t3 = dtm.now()
        self.updateE_calc_time =  t2 - t1
        self.updateE_exch_time = (t3 - t2) + (t1 - t0)
        self.updateE_comm_time =        0
        self.updateE_time      =  t3 - t0
        return evts

    def updateH_multi(self, wait=True):
        t0 = dtm.now()
        evts = []
        for i in xrange(self.num_devices):
            evts += self.fdtd_group[i].sync_fields(wait=False)
        wait_for_events(self, evts)
        t1 = dtm.now()
        if self.is_tfsf and not self.is_subfdtd:
            for tfsf in self.tfsfs:
                tfsf.tfsf_fdtd.updateH(wait=True)
        for priority in self.fdtd_group[0].engine.priority_order_h:
            for i in xrange(self.num_devices):
                if self.fdtd_group[i].engine.updates[priority] is not None:
                    evts += self.fdtd_group[i].engine.updates[priority](wait=False)
        if wait: wait_for_events(self, evts)
        t2 = dtm.now()
        if not self.is_subfdtd:
            self.exchange_fields_h_multi()
            for i in xrange(self.num_devices):
                self.fdtd_group[i].update_h_flags()
        t3 = dtm.now()
        self.updateH_calc_time =  t2 - t1
        self.updateH_exch_time = (t3 - t2) + (t1 - t0)
        self.updateH_comm_time =        0
        self.updateH_time      =  t3 - t0
        return evts

class FDTD_MPI:
    def __init__(self):
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size() - 1
        self.rank = self.comm.Get_rank()
        if self.size == 0:
            raise exception.FDTDError, "MPI Extension must be executed more than 2 hosts"
        if   self.rank == self.size:
            self.master = True
            self.worker = False
            self.first  = False
        elif self.rank ==  0:
            self.master = False
            self.worker = True
            self.first  = True
        else:
            self.master = False
            self.worker = True
            self.first  = False
        if self.rank == self.size-1:
            self.last   = True
        else:
            self.last   = False

        self.is_pml = False
        self.is_pbc = False
        self.pml_info = {}
        self.pbc_info = {}

        self.pbc_x = False
        self.nx_group, self.x_strt_group, self.x_stop_group = divide_xgrid(self.space_grid[0], self.size)

        self.min_ds = np.array([self.space_grid[i].min() for i in xrange(3)]).min()
        self.delta_t = self.dt*self.min_ds/c0
        self.x_SI_pts, self.y_SI_pts, self.z_SI_pts = \
            [np.array([0.] + list(np.array(self.space_grid[i]).cumsum())) for i in xrange(3)]

        import psutil as ps

        if self.worker:
            self.sub_nx = self.nx_group[self.rank]
            self.x_strt = self.x_strt_group[self.rank]
            self.x_stop = self.x_stop_group[self.rank]

            self.sub_space_grid = (self.space_grid[0][self.x_strt:self.x_stop], \
                self.space_grid[1], self.space_grid[2])

            #if self.engine_name == 'intel_cpu':
            #self.engine_name = 'intel_cpu_without_openmp'

            if self.from_external_engine:
                self.sub_fdtd = Basic_FDTD_3d(self.mode, self.sub_space_grid, dtype=self.dtype, engine=self.engines    , device_id=self.device_id, MPI_extension=False, is_subfdtd=True)
            else:
                self.sub_fdtd = Basic_FDTD_3d(self.mode, self.sub_space_grid, dtype=self.dtype, engine=self.engine_name, device_id=self.device_id, MPI_extension=False, is_subfdtd=True)
            if self.sub_fdtd.extension == 'multi':
                self.sub_fdtd.set_fields_multi(x_offset=self.x_SI_pts[self.x_strt], min_ds=self.min_ds)
                self.sub_fdtd.master = False
                self.sub_fdtd.first  = True
                self.sub_fdtd.last   = True
            if self.sub_fdtd.extension == 'single':
                if self.from_external_engine:
                    self.sub_fdtd.set_computing_engine(self.engines[0])
                else:
                    self.sub_fdtd.set_computing_engine(self.engine_name)
                self.sub_fdtd.set_fields_single(x_offset=self.x_SI_pts[self.x_strt], min_ds=self.min_ds)
                self.sub_fdtd.init_single()
                self.sub_fdtd.setup()
                self.sub_fdtd.master = False
                self.sub_fdtd.first  = True
                self.sub_fdtd.last   = True
            self.engines = self.sub_fdtd.engines

        real_dtype = comp_to_real(self.dtype)
        self.dtype_mpi = {np.float32:MPI.FLOAT, np.float64:MPI.DOUBLE}[real_dtype]
        self.e = 1; self.h = 1;

        name = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']
        self.ehs = self.ex, self.ey, self.ez, self.hx, self.hy, self.hz = \
        self.EHs = self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz = \
            [Fields_MPI(self, (self.nx, self.ny, self.nz), self.dtype, name=name[i]) for i in xrange(6)]

        if self.worker:
            set_coordinate(self)

            # MPI initialization for    blocking communications
            if   self.comm_mode == 'block':
                if not self.is_complex:
                    self.mpi_boundary = np.zeros((1,2,self.ny,self.nz), dtype=real_dtype)
                else:
                    self.mpi_boundary = np.zeros((2,2,self.ny,self.nz), dtype=real_dtype)
            # MPI initialization for nonblocking communications
            elif self.comm_mode == 'nonblock':
                self.init_exchange_nonblock()
            # MPI initialization for  overlaping communications
            elif self.comm_mode == 'overlap':
                buf_space_grid0 = (self.sub_space_grid[0][ 0:1], self.sub_space_grid[1], self.sub_space_grid[2])
                buf_space_grid1 = (self.sub_space_grid[0][-1: ], self.sub_space_grid[1], self.sub_space_grid[2])
                if isinstance(self.device_id, list) or isinstance(self.device_id, tuple):
                    buf_fdtd0 = Basic_FDTD_3d(self.mode, buf_space_grid0, dtype=self.dtype, device_id=self.device_id[ 0], is_subfdtd=True)
                    buf_fdtd1 = Basic_FDTD_3d(self.mode, buf_space_grid1, dtype=self.dtype, device_id=self.device_id[-1], is_subfdtd=True)
                else:
                    buf_fdtd0 = Basic_FDTD_3d(self.mode, buf_space_grid0, dtype=self.dtype, device_id=self.device_id    , is_subfdtd=True)
                    buf_fdtd1 = Basic_FDTD_3d(self.mode, buf_space_grid1, dtype=self.dtype, device_id=self.device_id    , is_subfdtd=True)

                self.buf_fdtd_group = [buf_fdtd0, buf_fdtd1]
                self.set_sub_engine('intel_cpu_without_openmp')
                #self.set_sub_engine('intel_cpu')
                self.init_exchange_overlap()

    def get_info_mpi(self):
        info = {}
        info['MODE'] = self.mode
        info['SHAPE'] = self.shape
        info['selfSIZE']  = self.n
        info['Distribution of x-axis grid'] = \
            [(self.x_strt_group[i], self.x_stop_group[i]) for i in xrange(self.size)]

        info['Uniform_Grid'] = self.is_uniform_grid
        info['CE Array'] = self.is_electric
        info['CH Array'] = self.is_magnetic
        info['Dispersive'] = self.is_dispersive

        info['PML'] = self.is_pml
        if self.is_pml:
            info['PML_apply'] = self.pml_info['pml_apply']
            info['PML_thick'] = self.pml_info['pml_thick']
            info['PML_PARAMETER'] = {'alpha':self.pml_info['alpha'], \
                                     'kappa':self.pml_info['kappa'], \
                                     'alpha exponent': self.pml_info['alpha_exponent'], \
                                     'sigma exponent': self.pml_info['sigma_exponent']}
        info['PBC'] = self.is_pbc
        if self.is_pbc:
            info['PBC_apply'] = self.pbc_info['pbc_apply']

        if self.master:
            for key in sorted(info.iterkeys()):
                print '%s: %s' % (key, info[key])
            for num, sinfo in enumerate(sinfos):
                print 'Assigned x grid(Device %s): (%s to %s)' % \
                      (num, self.x_strt_group[num], self.x_stop_group[num])

        if self.worker:
            if self.sub_fdtd.extension == 'single':
                sinfo = self.sub_fdtd.get_info_single()
            else:
                sinfo = self.sub_fdtd.get_info_multi()[1]
                items = ['SHAPE', \
                         'SIZE', \
                         'Computing Architecture', \
                         'Computing Device', \
                         'Memory size of Computing Device (MiB)', \
                         'Allocated Memory size (MiB)']
                for item in items:
                    for num, sinf in enumerate(sinfo):
                        print '%s(Device %s): %s' % (item, num, sinf[item])
        else:
            sinfo = None

        return info, sinfo

    def set_sub_engine(self, sub_engine):
        if not 'cpu' in sub_engine:
            raise NotImplementedError, 'buffer fdtd for MPI overlapping communication is implemented by using CPU only'
        bfdtd0, bfdtd1 = self.buf_fdtd_group
        for bfdtd in self.buf_fdtd_group:
            bfdtd.set_computing_engine(sub_engine)
            bfdtd.engine_name = bfdtd.engine.name
        bfdtd0.set_fields_single(x_offset=self.x_SI_pts[self.x_strt  ], min_ds=self.min_ds)
        bfdtd1.set_fields_single(x_offset=self.x_SI_pts[self.x_stop-1], min_ds=self.min_ds)
        for bfdtd in self.buf_fdtd_group:
            bfdtd.init_single()
            bfdtd.setup()

    def init_mpi(self, opt='fields'):
        if self.worker:
            self.sub_fdtd.init_multi(opt)

    def broadcast(self, value):
        self.comm.Bcast(value, root=self.size)
        return value

    def apply_PML_mpi(self, pml_apply, pml_thick, alpha0, kappa0, malpha, msigma, lnR0):
        from pml import PML
        self.is_pml = True
        self.pml_info['pml_apply'] = pml_apply
        self.pml_info['pml_thick'] = pml_thick
        self.pml_info['alpha'] = alpha0
        self.pml_info['kappa'] = kappa0
        self.pml_info['alpha_exponent'] = malpha
        self.pml_info['sigma_exponent'] = msigma
        self.pml_info['lnR0'] = lnR0
        if self.worker:
            if self.comm_mode == 'overlap':
                buf_pml_apply = {'x':'', 'y':pml_apply['y'], 'z':pml_apply['z']}
                buffdtd0 = self.buf_fdtd_group[ 0]
                buffdtd1 = self.buf_fdtd_group[-1]
                buffdtd0.apply_PML(buf_pml_apply, pml_thick, alpha0, kappa0, malpha, msigma, lnR0)
                buffdtd1.apply_PML(buf_pml_apply, pml_thick, alpha0, kappa0, malpha, msigma, lnR0)
            for i in xrange(self.sub_fdtd.num_devices):
                sub_pml_apply = {'x':'', 'y':pml_apply['y'], 'z':pml_apply['z']}
                if self.first and i == 0:
                    if '-' in pml_apply['x']:
                        sub_pml_apply['x'] += '-'
                if self.last  and i == self.sub_fdtd.num_devices - 1:
                    if '+' in pml_apply['x']:
                        sub_pml_apply['x'] += '+'
                subfdtd = self.sub_fdtd.fdtd_group[i]
                subfdtd.apply_PML(sub_pml_apply, pml_thick, alpha0, kappa0, malpha, msigma, lnR0)

    def apply_PBC_mpi(self, pbc_apply, klx=0., kly=0., klz=0.):
        from pbc import PBC, BBC
        self.is_pbc = True
        self.pbc_info['pbc_apply'] = pbc_apply
        self.pbc_info['klx'] = klx
        self.pbc_info['kly'] = kly
        self.pbc_info['klz'] = klz
        self.pbc_klx = klx
        self.sin_klx = np.sin(klx)
        self.cos_klx = np.cos(klx)
        if self.worker:
            if pbc_apply['x']:
                self.pbc_x = True
                if True in [self.first, self.last]:
                    if not self.is_complex:
                        self.pbc_boundary = np.zeros((1,2, self.ny, self.nz), dtype=comp_to_real(self.dtype))
                    else:
                        self.pbc_boundary = np.zeros((2,2, self.ny, self.nz), dtype=comp_to_real(self.dtype))
            else:
                self.pbc_x = False
            sub_pbc_apply = {'x':False, 'y':pbc_apply['y'], 'z':pbc_apply['z']}
            for i in xrange(self.sub_fdtd.num_devices):
                subfdtd = self.sub_fdtd.fdtd_group[i]
                subfdtd.apply_PBC(sub_pbc_apply, klx=klx, kly=kly, klz=klz)
            if self.comm_mode == 'overlap':
                for i in xrange(2):
                    buffdtd = self.buf_fdtd_group[i]
                    buffdtd.apply_PBC_single(sub_pbc_apply, klx=klx, kly=kly, klz=klz)

    def apply_TFSF_mpi(self, region, boundary={'x':'+-', 'y':'+-', 'z':'+-'}, is_oblique=True):
        from incident import TFSF_Boundary
        tfsf = TFSF_Boundary(self, region, boundary, is_oblique)
        self.tfsfs.append(tfsf)
        self.is_tfsf = True
        return tfsf.tfsf_fdtd

    def apply_TFSF1D_mpi(self, field, region, rot_vec, pol_vec, boundary={'x':'+-', 'y':'+-', 'z':'+-'}, material=None):
        from incident import TFSF_1D_Boundary
        tfsf = TFSF_1D_Boundary(self, field, region, rot_vec, pol_vec, boundary, material)
        self.tfsfs.append(tfsf)
        self.is_tfsf = True
        return tfsf.tfsf_inc

    def apply_direct_source_mpi(self, field, region):
        from incident import DirectSource
        return DirectSource(self, field, region)

    def apply_monochromatic_source_mpi(self, field, region, freq):
        from incident import MonochromaticSource
        return MonochromaticSource(self, field, region, freq)

    def apply_RFT_mpi(self, field, region, freq_domain):
        from rft import RFT
        return RFT(self, field, region, freq_domain)

    def set_structures_mpi(self, structures=None):
        if self.worker and structures is not None:
            self.sub_fdtd.set_structures_multi(structures)
            if self.comm_mode == 'overlap':
                self.buf_fdtd_group[0].set_structures_single(structures)
                self.buf_fdtd_group[1].set_structures_single(structures)

        for structure in structures:
            if structure.material.classification in ['dielectric', 'electric_dispersive', 'electromagnetic_dispersive']:
                self.is_electric = True
            if structure.material.classification in ['dimagnetic', 'magnetic_dispersive', 'electromagnetic_dispersive']:
                self.is_magnetic = True
            if structure.material.classification in ['electric_dispersive', 'magnetic_dispersive', 'electromagnetic_dispersive']:
                self.is_dispersive = True

        real_dtype = comp_to_real(self.dtype)
        if self.is_electric:
            name = ['ce1x', 'ce1y', 'ce1z', 'ce2x', 'ce2y', 'ce2z']
            self.ces = self.ce1x, self.ce1y, self.ce1z, self.ce2x, self.ce2y, self.ce2z = \
            self.cEs = self.cE1x, self.cE1y, self.cE1z, self.cE2x, self.cE2y, self.cE2z = \
                [Fields_MPI(self, self.shape, real_dtype, name=name[i]) for i in xrange(6)]
            self.ce1s = self.cE1s = self.ces[:3]
            self.ce2s = self.cE2s = self.ces[3:]
        if self.is_magnetic:
            name = ['ch1x', 'ch1y', 'ch1z', 'ch2x', 'ch2y', 'ch2z']
            self.chs = self.ch1x, self.ch1y, self.ch1z, self.ch2x, self.ch2y, self.ch2z = \
            self.cHs = self.cH1x, self.cH1y, self.cH1z, self.cH2x, self.cH2y, self.cH2z = \
                [Fields_MPI(self, self.shape, real_dtype, name=name[i]) for i in xrange(6)]
            self.ch1s = self.cH1s = self.chs[:3]
            self.ch2s = self.cH2s = self.chs[3:]

    def update_e_mpi(self):
        if self.worker:
            self.sub_fdtd.update_e()

    def update_h_mpi(self):
        if self.worker:
            self.sub_fdtd.update_h()

    # worker function: MUST NOT EXEC in MASTER RANK
    def exchange_fields_e_mpi(self):
        if self.pbc_x:
            if self.rank == 0:
                self.comm.Recv([self.pbc_boundary, self.dtype_mpi], source=self.size-1, tag=20)
                for i, part in enumerate(self.complex_parts):
                    self.sub_fdtd.fdtd_group[0].ey.__dict__[part][0,:,:] = self.pbc_boundary[i,0,:,:]
                    self.sub_fdtd.fdtd_group[0].ez.__dict__[part][0,:,:] = self.pbc_boundary[i,1,:,:]
            if self.rank == self.size-1:
                if not self.is_complex:
                    for i, part in enumerate(self.complex_parts):
                        self.pbc_boundary[i,0,:,:] = self.sub_fdtd.fdtd_group[-1].ey.__dict__[part][-1,:,:]
                        self.pbc_boundary[i,1,:,:] = self.sub_fdtd.fdtd_group[-1].ez.__dict__[part][-1,:,:]
                else:
                    ey_real = self.sub_fdtd.fdtd_group[-1].ey.real[-1,:,:]
                    ez_real = self.sub_fdtd.fdtd_group[-1].ez.real[-1,:,:]
                    ey_imag = self.sub_fdtd.fdtd_group[-1].ey.imag[-1,:,:]
                    ez_imag = self.sub_fdtd.fdtd_group[-1].ez.imag[-1,:,:]
                    # real part of ey
                    self.pbc_boundary[0,0,:,:] = + self.cos_klx*ey_real + self.sin_klx*ey_imag
                    # real part of ez
                    self.pbc_boundary[0,1,:,:] = + self.cos_klx*ez_real + self.sin_klx*ez_imag
                    # imag part of ey
                    self.pbc_boundary[1,0,:,:] = - self.sin_klx*ey_real + self.cos_klx*ey_imag
                    # imag part of ez
                    self.pbc_boundary[1,1,:,:] = - self.sin_klx*ez_real + self.cos_klx*ez_imag
                self.comm.Send([self.pbc_boundary, self.dtype_mpi], dest=0, tag=20)
        if   self.rank % 2 == 0:
            if self.rank != self.size-1:
                for i, part in enumerate(self.complex_parts):
                    self.mpi_boundary[i,0,:,:] = self.sub_fdtd.fdtd_group[-1].ey.__dict__[part][-1,:,:]
                    self.mpi_boundary[i,1,:,:] = self.sub_fdtd.fdtd_group[-1].ez.__dict__[part][-1,:,:]
                self.comm.Send([self.mpi_boundary, self.dtype_mpi],   dest=self.rank+1, tag=21)
            if self.rank != 0:
                self.comm.Recv([self.mpi_boundary, self.dtype_mpi], source=self.rank-1, tag=22)
                for i, part in enumerate(self.complex_parts):
                    self.sub_fdtd.fdtd_group[0].ey.__dict__[part][0,:,:] = self.mpi_boundary[i,0,:,:]
                    self.sub_fdtd.fdtd_group[0].ez.__dict__[part][0,:,:] = self.mpi_boundary[i,1,:,:]
        elif self.rank % 2 == 1:
            self.comm.Recv([self.mpi_boundary, self.dtype_mpi], source=self.rank-1, tag=21)
            for i, part in enumerate(self.complex_parts):
                self.sub_fdtd.fdtd_group[0].ey.__dict__[part][0,:,:] = self.mpi_boundary[i,0,:,:]
                self.sub_fdtd.fdtd_group[0].ez.__dict__[part][0,:,:] = self.mpi_boundary[i,1,:,:]
            if self.rank != self.size-1:
                for i, part in enumerate(self.complex_parts):
                    self.mpi_boundary[i,0,:,:] = self.sub_fdtd.fdtd_group[-1].ey.__dict__[part][-1,:,:]
                    self.mpi_boundary[i,1,:,:] = self.sub_fdtd.fdtd_group[-1].ez.__dict__[part][-1,:,:]
                self.comm.Send([self.mpi_boundary, self.dtype_mpi],   dest=self.rank+1, tag=22)
        for sfdtd in self.sub_fdtd.fdtd_group:
            sfdtd.ey.buff_updated_flag = True
            sfdtd.ez.buff_updated_flag = True
        evts = []
        for sfdtd in self.sub_fdtd.fdtd_group:
            evts += sfdtd.sync_fields(wait=False)
        wait_for_events(self, evts)

    # worker function: MUST NOT EXEC in MASTER RANK
    def exchange_fields_h_mpi(self):
        if self.pbc_x:
            if self.rank == 0:
                if not self.is_complex:
                    for i, part in enumerate(self.complex_parts):
                        self.pbc_boundary[i,0,:,:] = self.sub_fdtd.fdtd_group[0].hy.__dict__[part][0,:,:]
                        self.pbc_boundary[i,1,:,:] = self.sub_fdtd.fdtd_group[0].hz.__dict__[part][0,:,:]
                else:
                    hy_real = self.sub_fdtd.fdtd_group[-1].hy.real[-1,:,:]
                    hz_real = self.sub_fdtd.fdtd_group[-1].hz.real[-1,:,:]
                    hy_imag = self.sub_fdtd.fdtd_group[-1].hy.imag[-1,:,:]
                    hz_imag = self.sub_fdtd.fdtd_group[-1].hz.imag[-1,:,:]
                    # real part of hy
                    self.pbc_boundary[0,0,:,:] = + self.cos_klx*hy_real - self.sin_klx*hy_imag
                    # real part of hz
                    self.pbc_boundary[0,1,:,:] = + self.cos_klx*hz_real - self.sin_klx*hz_imag
                    # imag part of hy
                    self.pbc_boundary[1,0,:,:] = + self.sin_klx*hy_real + self.cos_klx*hy_imag
                    # imag part of hz
                    self.pbc_boundary[1,1,:,:] = + self.sin_klx*hz_real + self.cos_klx*hz_imag

                self.comm.Send([self.pbc_boundary, self.dtype_mpi], dest=self.size-1, tag=30)
            if self.rank == self.size-1:
                self.comm.Recv([self.pbc_boundary, self.dtype_mpi], source=0, tag=30)
                for i, part in enumerate(self.complex_parts):
                    self.sub_fdtd.fdtd_group[-1].hy.__dict__[part][-1,:,:] = self.pbc_boundary[i,0,:,:]
                    self.sub_fdtd.fdtd_group[-1].hz.__dict__[part][-1,:,:] = self.pbc_boundary[i,1,:,:]
        if   self.rank % 2 == 0:
            if self.rank != 0:
                for i, part in enumerate(self.complex_parts):
                    self.mpi_boundary[i,0,:,:] = self.sub_fdtd.fdtd_group[0].hy.__dict__[part][0,:,:]
                    self.mpi_boundary[i,1,:,:] = self.sub_fdtd.fdtd_group[0].hz.__dict__[part][0,:,:]
                self.comm.Send([self.mpi_boundary, self.dtype_mpi],   dest=self.rank-1, tag=31)
            if self.rank != self.size-1:
                self.comm.Recv([self.mpi_boundary, self.dtype_mpi], source=self.rank+1, tag=32)
                for i, part in enumerate(self.complex_parts):
                    self.sub_fdtd.fdtd_group[-1].hy.__dict__[part][-1,:,:] = self.mpi_boundary[i,0,:,:]
                    self.sub_fdtd.fdtd_group[-1].hz.__dict__[part][-1,:,:] = self.mpi_boundary[i,1,:,:]
        elif self.rank % 2 == 1:
            if self.rank != self.size-1:
                self.comm.Recv([self.mpi_boundary, self.dtype_mpi], source=self.rank+1, tag=31)
                for i, part in enumerate(self.complex_parts):
                    self.sub_fdtd.fdtd_group[-1].hy.__dict__[part][-1,:,:] = self.mpi_boundary[i,0,:,:]
                    self.sub_fdtd.fdtd_group[-1].hz.__dict__[part][-1,:,:] = self.mpi_boundary[i,1,:,:]
            for i, part in enumerate(self.complex_parts):
                self.mpi_boundary[i,0,:,:] = self.sub_fdtd.fdtd_group[0].hy.__dict__[part][0,:,:]
                self.mpi_boundary[i,1,:,:] = self.sub_fdtd.fdtd_group[0].hz.__dict__[part][0,:,:]
            self.comm.Send([self.mpi_boundary, self.dtype_mpi],   dest=self.rank-1, tag=32)
        for sfdtd in self.sub_fdtd.fdtd_group:
            sfdtd.hy.buff_updated_flag = True
            sfdtd.hz.buff_updated_flag = True
        evts = []
        for sfdtd in self.sub_fdtd.fdtd_group:
            evts += sfdtd.sync_fields(wait=False)
        wait_for_events(self, evts)

    # worker function: MUST NOT EXEC in MASTER RANK
    def init_exchange_nonblock(self):
        sfdtd = self.sub_fdtd
        real_dtype = comp_to_real(self.dtype)
        dsize = {np.float32:4, np.float64:8}[real_dtype]
        if not self.is_complex:
            self.buff_size  = 2*self.ny*self.nz*dsize
            self.buff_shape = (1,2,self.ny,self.nz)
        else:
            self.buff_size = 4*self.ny*self.nz*dsize
            self.buff_shape = (2,2,self.ny,self.nz)

        if   sfdtd.extension == 'multi':
            self.temp_send_e = sfdtd.fdtd_group[-1].temp_send_e
            self.temp_recv_h = sfdtd.fdtd_group[-1].temp_recv_h
            self.temp_recv_e = sfdtd.fdtd_group[ 0].temp_recv_e
            self.temp_send_h = sfdtd.fdtd_group[ 0].temp_send_h
        elif sfdtd.extension == 'single':
            self.temp_send_e, self.temp_send_h, self.temp_recv_e, self.temp_recv_h = \
                [np.zeros(self.buff_shape, dtype=real_dtype) for i in xrange(4)]
            if 'opencl' in sfdtd.engine.name:
                mf = sfdtd.fdtd_group[-1].engine.cl.mem_flags
                sfdtd.fdtd_group[-1].buff_send_e, sfdtd.fdtd_group[-1].buff_recv_h = \
                    [sfdtd.fdtd_group[-1].engine.cl.Buffer(sfdtd.fdtd_group[-1].engine.ctx, mf.READ_WRITE, self.buff_size) for i in xrange(2)]
                mf = sfdtd.fdtd_group[ 0].engine.cl.mem_flags
                sfdtd.fdtd_group[ 0].buff_send_h, sfdtd.fdtd_group[ 0].buff_recv_e = \
                    [sfdtd.fdtd_group[ 0].engine.cl.Buffer(sfdtd.fdtd_group[ 0].engine.ctx, mf.READ_WRITE, self.buff_size) for i in xrange(2)]
            elif 'cuda' in sfdtd.engine.name:
                sfdtd.fdtd_group[-1].buff_send_e, sfdtd.fdtd_group[-1].buff_recv_h = \
                    [sfdtd.fdtd_group[-1].engine.mem_alloc(self.buff_size) for i in xrange(2)]
                sfdtd.fdtd_group[ 0].buff_send_h, sfdtd.fdtd_group[ 0].buff_recv_e = \
                    [sfdtd.fdtd_group[ 0].engine.mem_alloc(self.buff_size) for i in xrange(2)]
            elif  'cpu' in self.engine_name:
                self.buff_send_e, self.buff_send_h, self.buff_recv_e, self.buff_recv_h = \
                    sfdtd.buff_send_e, sfdtd.buff_send_h, sfdtd.buff_recv_e, sfdtd.buff_recv_h = \
                        self.temp_send_e, self.temp_send_h, self.temp_recv_e, self.temp_recv_h

            sfdtd.engine.kernel_args['field_to_buf'] = {'ey':{'real':None, 'imag':None}, 'ez':{'real':None, 'imag':None}, \
                                                        'hy':{'real':None, 'imag':None}, 'hz':{'real':None, 'imag':None}  }
            sfdtd.engine.kernel_args['buf_to_field'] = {'ey':{'real':None, 'imag':None}, 'ez':{'real':None, 'imag':None}, \
                                                        'hy':{'real':None, 'imag':None}, 'hz':{'real':None, 'imag':None}  }

            if 'opencl' in sfdtd.engine_name:
                gs    = cl_global_size(self.ny*self.nz, sfdtd.fdtd_group[0].engine.ls)
            else:
                gs    = self.ny*self.nz
            for i, part in enumerate(sfdtd.complex_parts):
                sfdtd0 = sfdtd.fdtd_group[ 0]
                sfdtd1 = sfdtd.fdtd_group[-1]
                sfdtd1.engine.kernel_args['field_to_buf']['ey'][part] = [sfdtd1.engine.queue, (gs,), (sfdtd1.engine.ls,), \
                                                                         np.int32(1), np.int32(i*2+0), \
                                                                         np.int32(sfdtd1.nx), np.int32(sfdtd1.ny), np.int32(sfdtd1.nz), \
                                                                         sfdtd1.ey.__dict__[part].data, sfdtd1.buff_send_e]
                sfdtd1.engine.kernel_args['field_to_buf']['ez'][part] = [sfdtd1.engine.queue, (gs,), (sfdtd1.engine.ls,), \
                                                                         np.int32(1), np.int32(i*2+1), \
                                                                         np.int32(sfdtd1.nx), np.int32(sfdtd1.ny), np.int32(sfdtd1.nz), \
                                                                         sfdtd1.ez.__dict__[part].data, sfdtd1.buff_send_e]
                sfdtd0.engine.kernel_args['field_to_buf']['hy'][part] = [sfdtd0.engine.queue, (gs,), (sfdtd0.engine.ls,), \
                                                                         np.int32(0), np.int32(i*2+0), \
                                                                         np.int32(sfdtd0.nx), np.int32(sfdtd0.ny), np.int32(sfdtd0.nz), \
                                                                         sfdtd0.hy.__dict__[part].data, sfdtd0.buff_send_h]
                sfdtd0.engine.kernel_args['field_to_buf']['hz'][part] = [sfdtd0.engine.queue, (gs,), (sfdtd0.engine.ls,), \
                                                                         np.int32(0), np.int32(i*2+1), \
                                                                         np.int32(sfdtd0.nx), np.int32(sfdtd0.ny), np.int32(sfdtd0.nz), \
                                                                         sfdtd0.hz.__dict__[part].data, sfdtd0.buff_send_h]
                sfdtd0.engine.kernel_args['buf_to_field']['ey'][part] = [sfdtd0.engine.queue, (gs,), (sfdtd0.engine.ls,), \
                                                                         np.int32(0), np.int32(i*2+0), \
                                                                         np.int32(sfdtd0.nx), np.int32(sfdtd0.ny), np.int32(sfdtd0.nz), \
                                                                         sfdtd0.ey.__dict__[part].data, sfdtd0.buff_recv_e]
                sfdtd0.engine.kernel_args['buf_to_field']['ez'][part] = [sfdtd0.engine.queue, (gs,), (sfdtd0.engine.ls,), \
                                                                         np.int32(0), np.int32(i*2+1), \
                                                                         np.int32(sfdtd0.nx), np.int32(sfdtd0.ny), np.int32(sfdtd0.nz), \
                                                                         sfdtd0.ez.__dict__[part].data, sfdtd0.buff_recv_e]
                sfdtd1.engine.kernel_args['buf_to_field']['hy'][part] = [sfdtd1.engine.queue, (gs,), (sfdtd1.engine.ls,), \
                                                                         np.int32(1), np.int32(i*2+0), \
                                                                         np.int32(sfdtd1.nx), np.int32(sfdtd1.ny), np.int32(sfdtd1.nz), \
                                                                         sfdtd1.hy.__dict__[part].data, sfdtd1.buff_recv_h]
                sfdtd1.engine.kernel_args['buf_to_field']['hz'][part] = [sfdtd1.engine.queue, (gs,), (sfdtd1.engine.ls,), \
                                                                         np.int32(1), np.int32(i*2+1), \
                                                                         np.int32(sfdtd1.nx), np.int32(sfdtd1.ny), np.int32(sfdtd1.nz), \
                                                                         sfdtd1.hz.__dict__[part].data, sfdtd1.buff_recv_h]
            if 'opencl' in sfdtd.engine.name:
                sfdtd.engine.kernels['field_to_buf'] = sfdtd.engine.programs['get_set_data'].field_to_buf
                sfdtd.engine.kernels['buf_to_field'] = sfdtd.engine.programs['get_set_data'].buf_to_field
            elif 'cuda' in sfdtd.engine.name:
                sfdtd.engine.kernels['field_to_buf'] = sfdtd.engine.get_function(sfdtd.engine.programs['get_set_data'], 'field_to_buf')
                sfdtd.engine.kernels['buf_to_field'] = sfdtd.engine.get_function(sfdtd.engine.programs['get_set_data'], 'buf_to_field')
                sfdtd.engine.prepare(sfdtd.engine.kernels['field_to_buf'], sfdtd.engine.kernel_args['field_to_buf']['ey']['real'])
                sfdtd.engine.prepare(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['ey']['real'])

        if self.rank != 0:
            self.req_send_h = self.comm.Send_init([self.temp_send_h, self.dtype_mpi], self.rank-1, tag=11)
            self.req_recv_e = self.comm.Recv_init([self.temp_recv_e, self.dtype_mpi], self.rank-1, tag=10)
        else:
            self.req_send_h = self.comm.Send_init([self.temp_send_h, self.dtype_mpi], self.size-1, tag=11)
            self.req_recv_e = self.comm.Recv_init([self.temp_recv_e, self.dtype_mpi], self.size-1, tag=10)
        if self.rank != self.size-1:
            self.req_send_e = self.comm.Send_init([self.temp_send_e, self.dtype_mpi], self.rank+1, tag=10)
            self.req_recv_h = self.comm.Recv_init([self.temp_recv_h, self.dtype_mpi], self.rank+1, tag=11)
        else:
            self.req_send_e = self.comm.Send_init([self.temp_send_e, self.dtype_mpi],           0, tag=10)
            self.req_recv_h = self.comm.Recv_init([self.temp_recv_h, self.dtype_mpi],           0, tag=11)

    # worker function: MUST NOT EXEC in MASTER RANK
    def init_exchange_overlap(self):
        real_dtype = comp_to_real(self.dtype)
        dsize = {np.float32:4, np.float64:8}[real_dtype]
        bfdtd0, bfdtd1 = self.buf_fdtd_group
        sfdtd  = self.sub_fdtd
        sfdtd0 = self.sub_fdtd.fdtd_group[ 0]
        sfdtd1 = self.sub_fdtd.fdtd_group[-1]
        if not self.is_complex:
            self.buff_size  = 2*self.ny*self.nz*dsize
            self.buff_shape = (1, 2, self.ny, self.nz)
            self.temp_size  = 12*self.ny*self.nz*dsize
            self.temp_shape = (1, 6, 2, self.ny, self.nz)
        else:
            self.buff_size  = 4*self.ny*self.nz*dsize
            self.buff_shape = (2, 2, self.ny, self.nz)
            self.temp_size  = 24*self.ny*self.nz*dsize
            self.temp_shape = (2, 6, 2, self.ny, self.nz)
        self.temp_ehs0, self.temp_ehs1 = [np.zeros(self.temp_shape, dtype=real_dtype) for i in xrange(2)]

        if 'opencl' in sfdtd.engine_name:
            mf = sfdtd0.engine.cl.mem_flags
            self.buff_ehs0 = sfdtd0.engine.cl.Buffer(sfdtd0.engine.ctx, mf.READ_WRITE, self.temp_size)
            mf = sfdtd1.engine.cl.mem_flags
            self.buff_ehs1 = sfdtd1.engine.cl.Buffer(sfdtd1.engine.ctx, mf.READ_WRITE, self.temp_size)
            gs    = cl_global_size(2*self.ny*self.nz, sfdtd0.engine.ls)
        elif 'cuda' in sfdtd.engine_name:
            self.buff_ehs0 = sfdtd0.engine.mem_alloc(self.temp_size)
            self.buff_ehs1 = sfdtd1.engine.mem_alloc(self.temp_size)
            gs    = 2*self.ny*self.nz

        sfdtd0.engine.kernel_args['sub_sync_buf'] = {'+':{'real':None, 'imag':None}, '-':{'real':None, 'imag':None}}
        sfdtd1.engine.kernel_args['sub_sync_buf'] = {'+':{'real':None, 'imag':None}, '-':{'real':None, 'imag':None}}

        for i, part in enumerate(self.complex_parts):
            sfdtd0.engine.kernel_args['sub_sync_buf']['-'][part] = [sfdtd0.engine.queue, (gs,), (sfdtd0.engine.ls,), \
                                                                    np.int32(0), np.int32(i*12), \
                                                                    np.int32(sfdtd0.nx), np.int32(sfdtd0.ny), np.int32(sfdtd0.nz), \
                                                                    sfdtd0.ex.__dict__[part].data, \
                                                                    sfdtd0.ey.__dict__[part].data, \
                                                                    sfdtd0.ez.__dict__[part].data, \
                                                                    sfdtd0.hx.__dict__[part].data, \
                                                                    sfdtd0.hy.__dict__[part].data, \
                                                                    sfdtd0.hz.__dict__[part].data, \
                                                                    self.buff_ehs0]
            sfdtd1.engine.kernel_args['sub_sync_buf']['+'][part] = [sfdtd1.engine.queue, (gs,), (sfdtd1.engine.ls,), \
                                                                    np.int32(1), np.int32(i*12), \
                                                                    np.int32(sfdtd1.nx), np.int32(sfdtd1.ny), np.int32(sfdtd1.nz), \
                                                                    sfdtd1.ex.__dict__[part].data, \
                                                                    sfdtd1.ey.__dict__[part].data, \
                                                                    sfdtd1.ez.__dict__[part].data, \
                                                                    sfdtd1.hx.__dict__[part].data, \
                                                                    sfdtd1.hy.__dict__[part].data, \
                                                                    sfdtd1.hz.__dict__[part].data, \
                                                                    self.buff_ehs1]
        if 'opencl' in sfdtd.engine_name:
            sfdtd0.engine.kernels['sub_sync_buf'] = sfdtd0.engine.programs['get_set_data'].sub_sync_buf
            sfdtd1.engine.kernels['sub_sync_buf'] = sfdtd1.engine.programs['get_set_data'].sub_sync_buf
        elif 'cuda' in sfdtd.engine_name:
            sfdtd0.engine.kernels['sub_sync_buf'] = sfdtd0.engine.get_function(sfdtd0.engine.programs['get_set_data'], 'sub_sync_buf')
            sfdtd1.engine.kernels['sub_sync_buf'] = sfdtd1.engine.get_function(sfdtd1.engine.programs['get_set_data'], 'sub_sync_buf')
            sfdtd0.engine.prepare(sfdtd0.engine.kernels['sub_sync_buf'], sfdtd0.engine.kernel_args['sub_sync_buf']['-']['real'])
            sfdtd1.engine.prepare(sfdtd1.engine.kernels['sub_sync_buf'], sfdtd1.engine.kernel_args['sub_sync_buf']['+']['real'])

        self.temp_send_e, self.temp_send_h, self.temp_recv_e, self.temp_recv_h = \
            [np.zeros(self.buff_shape, dtype=real_dtype) for i in xrange(4)]
        if   sfdtd.extension == 'multi':
            self.temp_recv_h = sfdtd.fdtd_group[-1].temp_recv_h
            self.temp_recv_e = sfdtd.fdtd_group[ 0].temp_recv_e
        elif sfdtd.extension == 'single':
            if 'opencl' in sfdtd.engine.name:
                mf = sfdtd.fdtd_group[-1].engine.cl.mem_flags
                sfdtd.fdtd_group[-1].buff_send_e, sfdtd.fdtd_group[-1].buff_recv_h = \
                    [sfdtd.fdtd_group[-1].engine.cl.Buffer(sfdtd.fdtd_group[-1].engine.ctx, mf.READ_WRITE, self.buff_size) for i in xrange(2)]
                mf = sfdtd.fdtd_group[ 0].engine.cl.mem_flags
                sfdtd.fdtd_group[ 0].buff_send_h, sfdtd.fdtd_group[ 0].buff_recv_e = \
                    [sfdtd.fdtd_group[ 0].engine.cl.Buffer(sfdtd.fdtd_group[ 0].engine.ctx, mf.READ_WRITE, self.buff_size) for i in xrange(2)]
            elif 'cuda' in sfdtd.engine.name:
                sfdtd.fdtd_group[-1].buff_send_e, sfdtd.fdtd_group[-1].buff_recv_h = \
                    [sfdtd.fdtd_group[-1].engine.mem_alloc(self.buff_size) for i in xrange(2)]
                sfdtd.fdtd_group[ 0].buff_send_h, sfdtd.fdtd_group[ 0].buff_recv_e = \
                    [sfdtd.fdtd_group[ 0].engine.mem_alloc(self.buff_size) for i in xrange(2)]
            elif  'cpu' in self.engine_name:
                self.buff_send_e, self.buff_send_h, self.buff_recv_e, self.buff_recv_h = \
                    sfdtd.buff_send_e, sfdtd.buff_send_h, sfdtd.buff_recv_e, sfdtd.buff_recv_h = \
                        self.temp_send_e, self.temp_send_h, self.temp_recv_e, self.temp_recv_h

            sfdtd.engine.kernel_args['field_to_buf'] = {'ey':{'real':None, 'imag':None}, 'ez':{'real':None, 'imag':None}, \
                                                        'hy':{'real':None, 'imag':None}, 'hz':{'real':None, 'imag':None}  }
            sfdtd.engine.kernel_args['buf_to_field'] = {'ey':{'real':None, 'imag':None}, 'ez':{'real':None, 'imag':None}, \
                                                        'hy':{'real':None, 'imag':None}, 'hz':{'real':None, 'imag':None}  }

            if 'opencl' in sfdtd.engine_name:
                gs    = cl_global_size(self.ny*self.nz, sfdtd.fdtd_group[0].engine.ls)
            else:
                gs    = self.ny*self.nz
            for i, part in enumerate(sfdtd.complex_parts):
                sfdtd0 = sfdtd.fdtd_group[ 0]
                sfdtd1 = sfdtd.fdtd_group[-1]
                sfdtd1.engine.kernel_args['field_to_buf']['ey'][part] = [sfdtd1.engine.queue, (gs,), (sfdtd1.engine.ls,), \
                                                                         np.int32(1), np.int32(i*2+0), \
                                                                         np.int32(sfdtd1.nx), np.int32(sfdtd1.ny), np.int32(sfdtd1.nz), \
                                                                         sfdtd1.ey.__dict__[part].data, sfdtd1.buff_send_e]
                sfdtd1.engine.kernel_args['field_to_buf']['ez'][part] = [sfdtd1.engine.queue, (gs,), (sfdtd1.engine.ls,), \
                                                                         np.int32(1), np.int32(i*2+1), \
                                                                         np.int32(sfdtd1.nx), np.int32(sfdtd1.ny), np.int32(sfdtd1.nz), \
                                                                         sfdtd1.ez.__dict__[part].data, sfdtd1.buff_send_e]
                sfdtd0.engine.kernel_args['field_to_buf']['hy'][part] = [sfdtd0.engine.queue, (gs,), (sfdtd0.engine.ls,), \
                                                                         np.int32(0), np.int32(i*2+0), \
                                                                         np.int32(sfdtd0.nx), np.int32(sfdtd0.ny), np.int32(sfdtd0.nz), \
                                                                         sfdtd0.hy.__dict__[part].data, sfdtd0.buff_send_h]
                sfdtd0.engine.kernel_args['field_to_buf']['hz'][part] = [sfdtd0.engine.queue, (gs,), (sfdtd0.engine.ls,), \
                                                                         np.int32(0), np.int32(i*2+1), \
                                                                         np.int32(sfdtd0.nx), np.int32(sfdtd0.ny), np.int32(sfdtd0.nz), \
                                                                         sfdtd0.hz.__dict__[part].data, sfdtd0.buff_send_h]
                sfdtd0.engine.kernel_args['buf_to_field']['ey'][part] = [sfdtd0.engine.queue, (gs,), (sfdtd0.engine.ls,), \
                                                                         np.int32(0), np.int32(i*2+0), \
                                                                         np.int32(sfdtd0.nx), np.int32(sfdtd0.ny), np.int32(sfdtd0.nz), \
                                                                         sfdtd0.ey.__dict__[part].data, sfdtd0.buff_recv_e]
                sfdtd0.engine.kernel_args['buf_to_field']['ez'][part] = [sfdtd0.engine.queue, (gs,), (sfdtd0.engine.ls,), \
                                                                         np.int32(0), np.int32(i*2+1), \
                                                                         np.int32(sfdtd0.nx), np.int32(sfdtd0.ny), np.int32(sfdtd0.nz), \
                                                                         sfdtd0.ez.__dict__[part].data, sfdtd0.buff_recv_e]
                sfdtd1.engine.kernel_args['buf_to_field']['hy'][part] = [sfdtd1.engine.queue, (gs,), (sfdtd1.engine.ls,), \
                                                                         np.int32(1), np.int32(i*2+0), \
                                                                         np.int32(sfdtd1.nx), np.int32(sfdtd1.ny), np.int32(sfdtd1.nz), \
                                                                         sfdtd1.hy.__dict__[part].data, sfdtd1.buff_recv_h]
                sfdtd1.engine.kernel_args['buf_to_field']['hz'][part] = [sfdtd1.engine.queue, (gs,), (sfdtd1.engine.ls,), \
                                                                         np.int32(1), np.int32(i*2+1), \
                                                                         np.int32(sfdtd1.nx), np.int32(sfdtd1.ny), np.int32(sfdtd1.nz), \
                                                                         sfdtd1.hz.__dict__[part].data, sfdtd1.buff_recv_h]
            if 'opencl' in sfdtd.engine.name:
                sfdtd.engine.kernels['field_to_buf'] = sfdtd.engine.programs['get_set_data'].field_to_buf
                sfdtd.engine.kernels['buf_to_field'] = sfdtd.engine.programs['get_set_data'].buf_to_field
            elif 'cuda' in sfdtd.engine.name:
                sfdtd.engine.kernels['field_to_buf'] = sfdtd.engine.get_function(sfdtd.engine.programs['get_set_data'], 'field_to_buf')
                sfdtd.engine.kernels['buf_to_field'] = sfdtd.engine.get_function(sfdtd.engine.programs['get_set_data'], 'buf_to_field')
                sfdtd.engine.prepare(sfdtd.engine.kernels['field_to_buf'], sfdtd.engine.kernel_args['field_to_buf']['ey']['real'])
                sfdtd.engine.prepare(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['ey']['real'])

        if self.rank != 0:
            self.req_send_h = self.comm.Send_init([self.temp_send_h, self.dtype_mpi], self.rank-1, tag=11)
            self.req_recv_e = self.comm.Recv_init([self.temp_recv_e, self.dtype_mpi], self.rank-1, tag=10)
        else:
            self.req_send_h = self.comm.Send_init([self.temp_send_h, self.dtype_mpi], self.size-1, tag=11)
            self.req_recv_e = self.comm.Recv_init([self.temp_recv_e, self.dtype_mpi], self.size-1, tag=10)
        if self.rank != self.size-1:
            self.req_send_e = self.comm.Send_init([self.temp_send_e, self.dtype_mpi], self.rank+1, tag=10)
            self.req_recv_h = self.comm.Recv_init([self.temp_recv_h, self.dtype_mpi], self.rank+1, tag=11)
        else:
            self.req_send_e = self.comm.Send_init([self.temp_send_e, self.dtype_mpi],           0, tag=10)
            self.req_recv_h = self.comm.Recv_init([self.temp_recv_h, self.dtype_mpi],           0, tag=11)

    # worker function: MUST NOT EXEC in MASTER RANK
    def get_fields_mpi_nonblock(self, direction, wait=True):
        evts = []
        if   direction == '-':
            sfdtd = self.sub_fdtd.fdtd_group[0]
            if ('opencl' in sfdtd.engine.name) or ('cuda' in sfdtd.engine.name):
                if self.sub_fdtd.extension == 'multi':
                    self.sub_fdtd.send_h_multi()
                else:
                    if 'opencl' in sfdtd.engine.name:
                        for part in self.complex_parts:
                            evt = sfdtd.engine.kernels['field_to_buf'](*(sfdtd.engine.kernel_args['field_to_buf']['hy'][part]))
                            evts.append(evt)
                            evt = sfdtd.engine.kernels['field_to_buf'](*(sfdtd.engine.kernel_args['field_to_buf']['hz'][part]))
                            evts.append(evt)
                        evt = sfdtd.engine.cl.enqueue_read_buffer(sfdtd.engine.queue, sfdtd.buff_send_h, self.temp_send_h)
                        evts.append(evt)
                    elif 'cuda' in sfdtd.engine.name:
                        for part in self.complex_parts:
                            evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['field_to_buf'], sfdtd.engine.kernel_args['field_to_buf']['hy'][part])
                            evts.append(evt)
                            evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['field_to_buf'], sfdtd.engine.kernel_args['field_to_buf']['hz'][part])
                            evts.append(evt)
                        evt = sfdtd.engine.enqueue(sfdtd.engine.drv.memcpy_dtoh_async, [self.temp_send_h, sfdtd.buff_send_h], sfdtd.engine.stream)
                        evts.append(evt)
            elif  'cpu' in sfdtd.engine.name:
                for i, part in enumerate(self.complex_parts):
                    self.buff_send_h[i,0,:,:] = sfdtd.hy.__dict__[part][0,:,:]
                    self.buff_send_h[i,1,:,:] = sfdtd.hz.__dict__[part][0,:,:]
            if self.pbc_x and self.is_complex and self.first:
                nn = self.ny*self.nz
                self.pbc_boundary[:,:,:,:] = self.temp_send_h[:,:,:,:]
                # real part of hy
                self.temp_send_h[0,0,:,:] = + self.cos_klx*self.pbc_boundary[0,0,:,:] - self.sin_klx*self.pbc_boundary[1,0,:,:]
                # real part of hz
                self.temp_send_h[0,1,:,:] = + self.cos_klx*self.pbc_boundary[0,1,:,:] - self.sin_klx*self.pbc_boundary[1,1,:,:]
                # imag part of hy
                self.temp_send_h[1,0,:,:] = + self.sin_klx*self.pbc_boundary[0,0,:,:] + self.cos_klx*self.pbc_boundary[1,0,:,:]
                # imag part of hz
                self.temp_send_h[1,1,:,:] = + self.sin_klx*self.pbc_boundary[0,1,:,:] + self.cos_klx*self.pbc_boundary[1,1,:,:]

        elif direction == '+':
            sfdtd = self.sub_fdtd.fdtd_group[-1]
            if ('opencl' in sfdtd.engine.name) or ('cuda' in sfdtd.engine.name):
                if self.sub_fdtd.extension == 'multi':
                    self.sub_fdtd.send_e_multi()
                else:
                    if 'opencl' in sfdtd.engine.name:
                        for part in self.complex_parts:
                            evt = sfdtd.engine.kernels['field_to_buf'](*(sfdtd.engine.kernel_args['field_to_buf']['ey'][part]))
                            evts.append(evt)
                            evt = sfdtd.engine.kernels['field_to_buf'](*(sfdtd.engine.kernel_args['field_to_buf']['ez'][part]))
                            evts.append(evt)
                        evt = sfdtd.engine.cl.enqueue_read_buffer(sfdtd.engine.queue, sfdtd.buff_send_e, self.temp_send_e)
                        evts.append(evt)
                    elif 'cuda' in sfdtd.engine.name:
                        for part in self.complex_parts:
                            evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['field_to_buf'], sfdtd.engine.kernel_args['field_to_buf']['ey'][part])
                            evts.append(evt)
                            evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['field_to_buf'], sfdtd.engine.kernel_args['field_to_buf']['ez'][part])
                            evts.append(evt)
                        evt = sfdtd.engine.enqueue(sfdtd.engine.drv.memcpy_dtoh_async, [self.temp_send_e, sfdtd.buff_send_e], sfdtd.engine.stream)
                        evts.append(evt)
            elif  'cpu' in sfdtd.engine.name:
                for i, part in enumerate(self.complex_parts):
                    self.buff_send_e[i,0,:,:] = sfdtd.ey.__dict__[part][-1,:,:]
                    self.buff_send_e[i,1,:,:] = sfdtd.ez.__dict__[part][-1,:,:]
            if self.pbc_x and self.is_complex and self.last:
                self.pbc_boundary[:,:,:,:] = self.temp_send_e[:,:,:,:]
                # real part of ey
                self.temp_send_e[0,0,:,:] = + self.cos_klx*self.pbc_boundary[0,0,:,:] + self.sin_klx*self.pbc_boundary[1,0,:,:]
                # real part of ez
                self.temp_send_e[0,1,:,:] = + self.cos_klx*self.pbc_boundary[0,1,:,:] + self.sin_klx*self.pbc_boundary[1,1,:,:]
                # imag part of ey
                self.temp_send_e[1,0,:,:] = - self.sin_klx*self.pbc_boundary[0,0,:,:] + self.cos_klx*self.pbc_boundary[1,0,:,:]
                # imag part of ez
                self.temp_send_e[1,1,:,:] = - self.sin_klx*self.pbc_boundary[0,1,:,:] + self.cos_klx*self.pbc_boundary[1,1,:,:]
        if wait: wait_for_events(self, evts)
        return evts

    # worker function: MUST NOT EXEC in MASTER RANK
    def get_fields_mpi_overlap(self, direction, wait=True):
        evts = []
        if   direction == '-':
            sfdtd = self.sub_fdtd.fdtd_group[0]
            bfdtd = self.buf_fdtd_group[0]
            for part in self.complex_parts:
                if 'opencl' in sfdtd.engine.name:
                    evt = sfdtd.engine.kernels['sub_sync_buf'](*(sfdtd.engine.kernel_args['sub_sync_buf']['-'][part]))
                elif 'cuda' in sfdtd.engine.name:
                    evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['sub_sync_buf'], sfdtd.engine.kernel_args['sub_sync_buf']['-'][part])
                evts.append(evt)
            if 'opencl' in sfdtd.engine.name:
                evt = sfdtd.engine.cl.enqueue_read_buffer(sfdtd.engine.queue, self.buff_ehs0, self.temp_ehs0)
            elif 'cuda' in sfdtd.engine.name:
                evt = sfdtd.engine.enqueue(sfdtd.engine.drv.memcpy_dtoh_async, [self.temp_ehs0, self.buff_ehs0], sfdtd.engine.stream)
            evts.append(evt)
            if wait: wait_for_events(self, evts)
#            for i, part in enumerate(self.complex_parts):
#                for j, eh in enumerate(['ex', 'ey', 'ez', 'hx', 'hy', 'hz']):
#                    bfdtd.__dict__[eh].__dict__[part].data[:,:,:] = self.temp_ehs0[i,j,:,:,:]
        elif direction == '+':
            sfdtd = self.sub_fdtd.fdtd_group[-1]
            bfdtd = self.buf_fdtd_group[-1]
            for part in self.complex_parts:
                if 'opencl' in sfdtd.engine.name:
                    evt = sfdtd.engine.kernels['sub_sync_buf'](*(sfdtd.engine.kernel_args['sub_sync_buf']['+'][part]))
                elif 'cuda' in sfdtd.engine.name:
                    evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['sub_sync_buf'], sfdtd.engine.kernel_args['sub_sync_buf']['+'][part])
                evts.append(evt)
            if 'opencl' in sfdtd.engine.name:
                evt = sfdtd.engine.cl.enqueue_read_buffer(sfdtd.engine.queue, self.buff_ehs1, self.temp_ehs1)
            elif 'cuda' in sfdtd.engine.name:
                evt = sfdtd.engine.enqueue(sfdtd.engine.drv.memcpy_dtoh_async, [self.temp_ehs1, self.buff_ehs1], sfdtd.engine.stream)
            evts.append(evt)
            if wait: wait_for_events(self, evts)
#            for i, part in enumerate(self.complex_parts):
#                for j, eh in enumerate(['ex', 'ey', 'ez', 'hx', 'hy', 'hz']):
#                    bfdtd.__dict__[eh].__dict__[part].data[:,:,:] = self.temp_ehs1[i,j,:,:,:]
        return evts

    # worker function: MUST NOT EXEC in MASTER RANK
    def get_fields_to_buff_fdtd(self, direction):
        if   direction == '-':
            sfdtd = self.sub_fdtd.fdtd_group[0]
            bfdtd = self.buf_fdtd_group[0]
            for i, part in enumerate(self.complex_parts):
                for j, eh in enumerate(['ex', 'ey', 'ez', 'hx', 'hy', 'hz']):
                    bfdtd.__dict__[eh].__dict__[part].data[:,:,:] = self.temp_ehs0[i,j,:,:,:]
        elif direction == '+':
            sfdtd = self.sub_fdtd.fdtd_group[-1]
            bfdtd = self.buf_fdtd_group[-1]
            for i, part in enumerate(self.complex_parts):
                for j, eh in enumerate(['ex', 'ey', 'ez', 'hx', 'hy', 'hz']):
                    bfdtd.__dict__[eh].__dict__[part].data[:,:,:] = self.temp_ehs1[i,j,:,:,:]

    # worker function: MUST NOT EXEC in MASTER RANK
    def set_fields_mpi_nonblock(self, direction, wait=True):
        evts = []
        if   direction == '-':
            sfdtd = self.sub_fdtd.fdtd_group[0]
            if ('opencl' in sfdtd.engine.name) or ('cuda' in sfdtd.engine.name):
                if self.sub_fdtd.extension == 'multi':
                    self.sub_fdtd.recv_e_multi()
                else:
                    if 'opencl' in sfdtd.engine.name:
                        evt = sfdtd.engine.cl.enqueue_write_buffer(sfdtd.engine.queue, sfdtd.buff_recv_e, self.temp_recv_e)
                        evts.append(evt)
                        for part in self.complex_parts:
                            evt = sfdtd.engine.kernels['buf_to_field'](*(sfdtd.engine.kernel_args['buf_to_field']['ey'][part]))
                            evts.append(evt)
                            evt = sfdtd.engine.kernels['buf_to_field'](*(sfdtd.engine.kernel_args['buf_to_field']['ez'][part]))
                            evts.append(evt)
                    elif 'cuda' in sfdtd.engine.name:
                        evt = sfdtd.engine.enqueue(sfdtd.engine.drv.memcpy_htod_async, [sfdtd.buff_recv_e, self.temp_recv_e], sfdtd.engine.stream)
                        evts.append(evt)
                        for part in self.complex_parts:
                            evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['ey'][part])
                            evts.append(evt)
                            evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['ez'][part])
                            evts.append(evt)
            elif  'cpu' in sfdtd.engine.name:
                for i, part in enumerate(self.complex_parts):
                    sfdtd.ey.__dict__[part][0,:,:] = self.buff_recv_e[i,0,:,:]
                    sfdtd.ez.__dict__[part][0,:,:] = self.buff_recv_e[i,1,:,:]

        elif direction == '+':
            sfdtd = self.sub_fdtd.fdtd_group[-1]
            if ('opencl' in sfdtd.engine.name) or ('cuda' in sfdtd.engine.name):
                if self.sub_fdtd.extension == 'multi':
                    self.sub_fdtd.recv_h_multi()
                else:
                    if 'opencl' in sfdtd.engine.name:
                        evt = sfdtd.engine.cl.enqueue_write_buffer(sfdtd.engine.queue, sfdtd.buff_recv_h, self.temp_recv_h)
                        evts.append(evt)
                        for part in self.complex_parts:
                            evt = sfdtd.engine.kernels['buf_to_field'](*(sfdtd.engine.kernel_args['buf_to_field']['hy'][part]))
                            evts.append(evt)
                            evt = sfdtd.engine.kernels['buf_to_field'](*(sfdtd.engine.kernel_args['buf_to_field']['hz'][part]))
                            evts.append(evt)
                    elif 'cuda' in sfdtd.engine.name:
                        evt = sfdtd.engine.enqueue(sfdtd.engine.drv.memcpy_htod_async, [sfdtd.buff_recv_h, self.temp_recv_h], sfdtd.engine.stream)
                        evts.append(evt)
                        for part in self.complex_parts:
                            evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['hy'][part])
                            evts.append(evt)
                            evt = sfdtd.engine.enqueue_kernel(sfdtd.engine.kernels['buf_to_field'], sfdtd.engine.kernel_args['buf_to_field']['hz'][part])
                            evts.append(evt)
            elif  'cpu' in sfdtd.engine.name:
                for i, part in enumerate(self.complex_parts):
                    sfdtd.hy.__dict__[part][-1,:,:] = self.buff_recv_h[i,0,:,:]
                    sfdtd.hz.__dict__[part][-1,:,:] = self.buff_recv_h[i,1,:,:]
        if wait: wait_for_events(self, evts)
        return evts

    # worker function: MUST NOT EXEC in MASTER RANK
    def set_fields_mpi_overlap(self, direction, wait=True):
        return self.set_fields_mpi_nonblock(direction, wait=wait)

    # worker function: MUST NOT EXEC in MASTER RANK
    def exchange_fields_e_mpi_nonblock_start(self):
        if not self.first:
            self.req_recv_e.Start()
        if not self.last:
            self.req_send_e.Start()
        if self.pbc_x:
            if self.first:
                self.req_recv_e.Start()
            if self.last:
                self.req_send_e.Start()

    # worker function: MUST NOT EXEC in MASTER RANK
    def exchange_fields_e_mpi_nonblock_wait(self):
        if self.pbc_x:
            if self.first:
                self.req_recv_e.Wait()
            if self.last:
                self.req_send_e.Wait()
        if not self.first:
            self.req_recv_e.Wait()
        if not self.last:
            self.req_send_e.Wait()

    # worker function: MUST NOT EXEC in MASTER RANK
    def exchange_fields_h_mpi_nonblock_start(self):
        if not self.first:
            self.req_send_h.Start()
        if not self.last:
            self.req_recv_h.Start()
        if self.pbc_x:
            if self.first:
                self.req_send_h.Start()
            if self.last:
                self.req_recv_h.Start()

    # worker function: MUST NOT EXEC in MASTER RANK
    def exchange_fields_h_mpi_nonblock_wait(self):
        if self.pbc_x:
            if self.first:
                self.req_send_h.Wait()
            if self.last:
                self.req_recv_h.Wait()
        if not self.first:
            self.req_send_h.Wait()
        if not self.last:
            self.req_recv_h.Wait()

    # worker function: MUST NOT EXEC in MASTER RANK
    def updateE_mpi_block(self, wait=True):
        t0 = dtm.now()
        self.sub_fdtd.updateE()
        t1 = dtm.now()
        if self.sub_fdtd.extension == 'multi':
            self.sub_fdtd.exchange_fields_e_multi()
        for sfdtd in self.sub_fdtd.fdtd_group:
            sfdtd.update_e_flags()
        t2 = dtm.now()
        self.exchange_fields_e_mpi()
        t3 = dtm.now()
        self.updateE_calc_time = t1 - t0
        self.updateE_buff_time =  0.
        self.updateE_exch_time = t2 - t1
        self.updateE_comm_time = t3 - t2
        self.updateE_time      = t4 - t0

    # worker function: MUST NOT EXEC in MASTER RANK
    def updateE_mpi_nonblock(self, wait=True):
        t0 = dtm.now()
        self.sub_fdtd.updateE()
        t1 = dtm.now()
        self.get_fields_mpi_nonblock('+')
        t2 = dtm.now()
        self.exchange_fields_e_mpi_nonblock_start()
#        if self.sub_fdtd.extension == 'multi':
#            self.sub_fdtd.comm_e_multi()
        self.exchange_fields_e_mpi_nonblock_wait()
        t3 = dtm.now()
        self.set_fields_mpi_nonblock('-')
        t4 = dtm.now()
        self.updateE_calc_time =  t1 - t0
        self.updateE_buff_time =  0.
        self.updateE_exch_time = (t2 - t1) + (t4 - t3)
        self.updateE_comm_time =  t3 - t2
        self.updateE_time      =  t4 - t0

    # worker function: MUST NOT EXEC in MASTER RANK
    def updateE_mpi_overlap(self, wait=True):
        t0 = dtm.now()
        evts_get = self.get_fields_mpi_overlap('+', wait=False)
        t1 = dtm.now()
        evts = self.sub_fdtd.updateE(wait=False)
        t2 = dtm.now()
        wait_for_events(self, evts_get)
        self.get_fields_to_buff_fdtd('+')
        self.buf_fdtd_group[-1].updateE()
        t3 = dtm.now()
        if self.pbc_x and self.last and self.is_complex:
            # real part of ey
            self.temp_send_e[0,0,:,:] = + self.cos_klx*self.buf_fdtd_group[-1].ey.real[-1,:,:] \
                                        + self.sin_klx*self.buf_fdtd_group[-1].ey.imag[-1,:,:]
            # real part of ez
            self.temp_send_e[0,1,:,:] = + self.cos_klx*self.buf_fdtd_group[-1].ez.real[-1,:,:] \
                                        + self.sin_klx*self.buf_fdtd_group[-1].ez.imag[-1,:,:]
            # imag part of ey
            self.temp_send_e[1,0,:,:] = - self.sin_klx*self.buf_fdtd_group[-1].ey.real[-1,:,:] \
                                        + self.cos_klx*self.buf_fdtd_group[-1].ey.imag[-1,:,:]
            # imag part of ez
            self.temp_send_e[1,1,:,:] = - self.sin_klx*self.buf_fdtd_group[-1].ez.real[-1,:,:] \
                                        + self.cos_klx*self.buf_fdtd_group[-1].ez.imag[-1,:,:]
        else:
            for i, part in enumerate(self.complex_parts):
                self.temp_send_e[i,0,:,:] = self.buf_fdtd_group[-1].ey.__dict__[part][-1,:,:]
                self.temp_send_e[i,1,:,:] = self.buf_fdtd_group[-1].ez.__dict__[part][-1,:,:]
        t4 = dtm.now()
        self.exchange_fields_e_mpi_nonblock_start()
        if self.sub_fdtd.extension == 'multi':
            self.sub_fdtd.send_e_multi()
#            self.sub_fdtd.comm_e_multi()
        self.exchange_fields_e_mpi_nonblock_wait()
        t5 = dtm.now()
        evts += self.set_fields_mpi_overlap('-', wait=False)
        wait_for_events(self, evts)
        t6 = dtm.now()
        self.updateE_calc_time =  t2 - t1
        self.updateE_buff_time =  t3 - t2
        self.updateE_exch_time = (t1 - t0) + (t6 - t5)
        self.updateE_comm_time =  t5 - t4
        self.updateE_time      =  t6 - t0

    # worker function: MUST NOT EXEC in MASTER RANK
    def updateH_mpi_block(self, wait=True):
        t0 = dtm.now()
        self.sub_fdtd.updateH()
        t1 = dtm.now()
        if self.sub_fdtd.extension == 'multi':
            self.sub_fdtd.exchange_fields_h_multi()
        for sfdtd in self.sub_fdtd.fdtd_group:
            sfdtd.update_h_flags()
        t2 = dtm.now()
        self.exchange_fields_h_mpi()
        t3 = dtm.now()
        self.updateH_calc_time = t1 - t0
        self.updateH_buff_time =  0.
        self.updateH_exch_time = t2 - t1
        self.updateH_comm_time = t3 - t2
        self.updateH_time      = t4 - t0

    # worker function: MUST NOT EXEC in MASTER RANK
    def updateH_mpi_nonblock(self, wait=True):
        t0 = dtm.now()
        self.sub_fdtd.updateH()
        t1 = dtm.now()
        self.get_fields_mpi_nonblock('-')
        t2 = dtm.now()
        self.exchange_fields_h_mpi_nonblock_start()
#        if self.sub_fdtd.extension == 'multi':
#            self.sub_fdtd.comm_h_multi()
        self.exchange_fields_h_mpi_nonblock_wait()
        t3 = dtm.now()
        self.set_fields_mpi_nonblock('+')
        t4 = dtm.now()
        self.updateH_calc_time =  t1 - t0
        self.updateH_buff_time =  0.
        self.updateH_exch_time = (t2 - t1) + (t4 - t3)
        self.updateH_comm_time =  t3 - t2
        self.updateH_time      =  t4 - t0

    # worker function: MUST NOT EXEC in MASTER RANK
    def updateH_mpi_overlap(self, wait=True):
        t0 = dtm.now()
        evts_get = self.get_fields_mpi_overlap('-', wait=False)
        t1 = dtm.now()
        evts = self.sub_fdtd.updateH(wait=False)
        t2 = dtm.now()
        wait_for_events(self, evts_get)
        self.get_fields_to_buff_fdtd('-')
        self.buf_fdtd_group[ 0].updateH()
        t3 = dtm.now()
        if self.pbc_x and self.first and self.is_complex:
            # real part of hy
            self.temp_send_h[0,0,:,:] = + self.cos_klx*self.buf_fdtd_group[ 0].hy.real[ 0,:,:] \
                                        - self.sin_klx*self.buf_fdtd_group[ 0].hy.imag[ 0,:,:]
            # real part of hz
            self.temp_send_h[0,1,:,:] = + self.cos_klx*self.buf_fdtd_group[ 0].hz.real[ 0,:,:] \
                                        - self.sin_klx*self.buf_fdtd_group[ 0].hz.imag[ 0,:,:]
            # imag part of hy
            self.temp_send_h[1,0,:,:] = + self.cos_klx*self.buf_fdtd_group[ 0].hy.real[ 0,:,:] \
                                        + self.sin_klx*self.buf_fdtd_group[ 0].hy.imag[ 0,:,:]
            # imag part of hz
            self.temp_send_h[1,1,:,:] = + self.cos_klx*self.buf_fdtd_group[ 0].hz.real[ 0,:,:] \
                                        + self.sin_klx*self.buf_fdtd_group[ 0].hz.imag[ 0,:,:]
        else:
            for i, part in enumerate(self.complex_parts):
                self.temp_send_h[i,0,:,:] = self.buf_fdtd_group[ 0].hy.__dict__[part][ 0,:,:]
                self.temp_send_h[i,1,:,:] = self.buf_fdtd_group[ 0].hz.__dict__[part][ 0,:,:]
        t4 = dtm.now()
        self.exchange_fields_h_mpi_nonblock_start()
        if self.sub_fdtd.extension == 'multi':
            self.sub_fdtd.send_h_multi()
#            self.sub_fdtd.comm_h_multi()
        self.exchange_fields_h_mpi_nonblock_wait()
        t5 = dtm.now()
        evts += self.set_fields_mpi_overlap('+', wait=False)
        wait_for_events(self, evts)
        t6 = dtm.now()
        self.updateH_calc_time =  t2 - t1
        self.updateH_buff_time =  t3 - t2
        self.updateH_exch_time = (t1 - t0) + (t6 - t5)
        self.updateH_comm_time =  t5 - t4
        self.updateH_time      =  t6 - t0

    def updateE_mpi(self, wait=True):
        evts = []
        if self.worker:
            for sfdtd in self.sub_fdtd.fdtd_group:
                evts += sfdtd.sync_fields(wait=False)
            if self.is_tfsf:
                for tfsf in self.tfsfs:
                    tfsf.tfsf_fdtd.updateE()
            wait_for_events(self, evts)
            if   self.comm_mode == 'block':
                self.updateE_mpi_block()
            elif self.comm_mode == 'nonblock':
                self.updateE_mpi_nonblock()
            elif self.comm_mode == 'overlap':
                self.updateE_mpi_overlap()
            for sfdtd in self.sub_fdtd.fdtd_group:
                sfdtd.update_e_flags()
        self.comm.barrier()

    def updateH_mpi(self, wait=True):
        evts = []
        if self.worker:
            for sfdtd in self.sub_fdtd.fdtd_group:
                evts += sfdtd.sync_fields(wait=False)
            if self.is_tfsf:
                for tfsf in self.tfsfs:
                    tfsf.tfsf_fdtd.updateH()
            wait_for_events(self, evts)
            if   self.comm_mode == 'block':
                self.updateH_mpi_block()
            elif self.comm_mode == 'nonblock':
                self.updateH_mpi_nonblock()
            elif self.comm_mode == 'overlap':
                self.updateH_mpi_overlap()
            for sfdtd in self.sub_fdtd.fdtd_group:
                sfdtd.update_h_flags()
        self.comm.barrier()

class Basic_FDTD_2d(FDTD_single_device):
    def __init__(self, mode, space_grid, dtype=np.float32, engine='nvidia_opencl', device_id=0):
        # input variables
        self.mode = mode
        self.space_grid  = space_grid
        self.nx, self.ny = [space_grid[i].size + 1 for i in xrange(2)]

        self.dt     = .5
        self.shape  = (self.nx, self.ny)
        self.n      = self.nx*self.ny
        self.nbytes = 0
        self.dtype  = dtype
        self.fields_list = []
        self.sources = []
        self.tfsfs  = []
        self.rfts   = []

        if   self.dtype in [np.float32, np.float64]:
            self.is_complex = False
            self.complex_parts = ['real']
        elif self.dtype in [np.complex64, np.complex128]:
            self.is_complex = True
            self.complex_parts = ['real', 'imag']
        else:
            raise TypeError, 'Fields\' dtype of basic_fdtd class instance must be in [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128]'
        self.rdtype = {np.float32:np.float32, np.float64:np.float64, np.complex64:np.float32, np.complex128:np.float64}[dtype]

        self.extension = 'single'
        self.device_id = device_id

        if isinstance(engine, str):
            check_value('engine', engine, engines)
            self.engine_name = engine
            self.from_external_engine = False
        elif isinstance(engine, list) or isinstance(engine, tuple):
            from engine import engine as eng, subengine
            check_type('engine', engine, (list, tuple), eng)
            self.engine_name = engine[0].name
            self.engines = engine
            self.engine = subengine(engine[0])
            self.from_external_engine = True
        else:
            from engine import engine as eng, subengine
            check_type('engine', engine, eng)
            self.engine_name = engine.name
            self.engines = [engine]
            self.engine = subengine(engine)
            self.from_external_engine = True

        self.is_uniform_grid = True
        self.is_electric = False
        self.is_magnetic = False
        self.is_dispersive = False
        self.is_subfdtd  = False
        self.is_tfsf     = False
        self.is_pml      = False
        self.is_pbc      = False
        for grids in space_grid:
            if grids.std()/grids.min() > 1e-5: self.is_uniform_grid = False

        # check input variables
        if len(space_grid) != 2:
            raise IndexError, '2d-fdtd space must have 2d-space grids. Please check the input variable \'space_grid\''
        if self.extension not in ['single', 'multi', 'MPI']:
            raise TypeError, 'Device option must be in [\'single\', \'multi\', \'MPI\'].'
        if self.extension in ['multi', 'MPI']:
            print 'Warning: 2D-FDTD class doesn\'t support Multi, MPI device option. Single device option is selected.'
            self.extension = 'single'

        option_fdtd = {'single':FDTD_single_device, 'multi':FDTD_multi_devices, 'mpi':FDTD_MPI}[self.extension]
        option_fdtd.__init__(self)

    def init(self, opt='fields'):
        self.init_single(opt)

    def set_structures(self, structures=None):
        self.set_structures_single(structures)

    def apply_PML(self, pml_apply, pml_thick={'x':(10,10),'y':(10,10),'z':(10,10)}, alpha0=0., kappa0=1., malpha=4, msigma=4, lnR0=-16.):
        self.is_pml = True
        self.pml_apply = pml_apply
        self.apply_PML_single(pml_apply, pml_thick, alpha0, kappa0, malpha, msigma, lnR0)

    def apply_PBC(self, pbc_apply, klx=0., kly=0., klz=0.):
        self.is_pbc = True
        self.pbc_apply = pbc_apply
        self.bbc_coeff = (klx, kly, klz)
        self.apply_PBC_single(pbc_apply, klx=klx, kly=kly, klz=klz)

    def apply_TFSF(self, region, boundary={'x':'+-', 'y':'+-', 'z':'+-'}):
        from incident import TFSF_Boundary
        tfsf = TFSF_Boundary(self, region, boundary)
        self.tfsfs.append(tfsf)
        self.is_tfsf = True
        return tfsf.tfsf_fdtd

    def apply_TFSF1D(self, field, region, rot_vec, pol_vec, boundary={'x':'+-', 'y':'+-', 'z':'+-'}, material=None):
        from incident import TFSF_1D_Boundary
        tfsf = TFSF_1D_Boundary(self, field, region, rot_vec, pol_vec, boundary, material)
        self.tfsfs.append(tfsf)
        self.is_tfsf = True
        return tfsf.tfsf_inc

    def apply_direct_source(self, field, region):
        from incident import DirectSource
        return DirectSource(self, field, region)

    def apply_monochromatic_source(self, field, region, freq):
        from incident import MonochromaticSource
        return DirectSource(self, field, region, freq)

    def apply_RFT(self, field, region, freq_domain):
        from rft import RFT
        return RFT(self, field, region, freq_domain)

    def update_e(self, wait=True):
        return self.update_e_single(wait=wait)

    def update_h(self, wait=True):
        return self.update_h_single(wait=wait)

    def updateE(self, wait=True):
        return self.updateE_single(wait=wait)

    def updateH(self, wait=True):
        return self.updateH_single(wait=wait)

class Basic_FDTD_3d(FDTD_single_device, FDTD_multi_devices, FDTD_MPI):
    def __init__(self, mode, space_grid, dtype=np.float32, engine='nvidia_opencl', device_id=0, MPI_extension=False, is_subfdtd=False):
        # input variables
        check_type('space_grid', space_grid, (tuple, list), np.ndarray)
        check_type('is_subfdtd', is_subfdtd, bool)
        check_value('dtype', dtype, numpy_float_types + numpy_complex_types)
        check_value('MPI_extension', MPI_extension, (False, True, 'block', 'nonblock', 'overlap'))
        if isinstance(device_id, tuple) or isinstance(device_id, list):
            check_type('device_id', device_id, (tuple, list), int_types)
        else:
            check_type('device_id', device_id, int_types)
        if isinstance(engine, str):
            check_value('engine', engine, engines)
            self.engine_name = engine
            self.from_external_engine = False
        elif (isinstance(engine, list)) or (isinstance(engine, tuple)):
            from engine import engine as eng, subengine
            check_type('engine', engine, (list, tuple), eng)
            self.engine_name = engine[0].name
            self.engines = engine
            self.from_external_engine = True
        else:
            from engine import engine as eng, subengine
            check_type('engine', engine, eng)
            self.engine_name = engine.name
            self.engines = [engine]
            self.from_external_engine = True

        self.mode = mode
        self.device_id = device_id
        self.space_grid = space_grid
        self.nx, self.ny, self.nz = [space_grid[i].size + 1 for i in xrange(3)]

        self.shape = (self.nx, self.ny, self.nz)
        self.n     = self.nx*self.ny*self.nz
        self.dt    = .5
        self.dtype  = dtype
        self.sources = []
        self.tfsfs = []
        self.rfts = []

        self.nbytes = 0
        self.fields_list = []

        if   self.dtype in numpy_float_types:
            self.is_complex = False
            self.complex_parts = ['real']
        elif self.dtype in numpy_complex_types:
            self.is_complex = True
            self.complex_parts = ['real', 'imag']
        else:
            raise TypeError, '"Fields" dtype of basic_fdtd class instance must be in [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128]'
        self.rdtype = {np.float32:np.float32, np.float64:np.float64, np.complex64:np.float32, np.complex128:np.float64}[dtype]

        self.is_subfdtd = is_subfdtd
        self.is_uniform_grid = True
        self.is_electric = False
        self.is_magnetic = False
        self.is_dispersive = False
        self.is_tfsf    = False
        self.is_pml     = False
        self.is_pbc     = False
        if len(space_grid) != 3:
            raise IndexError, '2d-fdtd space must have 2d-space grids. Please check the input variable "space_grid"'

        self.is_uniform_grid = check_uniform_grid(space_grid)

        if MPI_extension in (True, 'block', 'nonblock', 'overlap'):
            self.extension = 'mpi'
            self.comm_mode = MPI_extension
            if MPI_extension == True:
                self.comm_mode = 'overlap'
            if self.from_external_engine is True:
                self.num_devices = len(self.engines)
                FDTD_MPI.__init__(self)
            else:
                if isinstance(device_id, tuple) or isinstance(device_id, list):
                    self.num_devices = len(device_id)
                else:
                    self.num_devices = 1
                FDTD_MPI.__init__(self)
        else:
            self.comm_mode = MPI_extension
            if self.from_external_engine is True:
                self.num_devices = len(self.engines)
                if self.num_devices == 1:
                    self.extension = 'single'
                    FDTD_single_device.__init__(self)
                else:
                    self.extension = 'multi'
                    FDTD_multi_devices.__init__(self)
            else:
                if isinstance(device_id, tuple) or isinstance(device_id, list):
                    if len(device_id) >= 2:
                        if 'cpu' in self.engine_name:
                            raise NotImplementedError, 'ComputingEngine %s is not implemented with multi_devices' % self.engine_name
                        self.extension = 'multi'
                        self.num_devices = len(device_id)
                        FDTD_multi_devices.__init__(self)
                    else:
                        self.extension = 'single'
                        self.num_devices = 1
                        self.device_id = device_id[0]
                        FDTD_single_device.__init__(self)
                else:
                    self.extension = 'single'
                    self.num_devices = 1
                    FDTD_single_device.__init__(self)

        self.apply_PML_list    = {'single':self.apply_PML_single   , 'multi':self.apply_PML_multi   , 'mpi':self.apply_PML_mpi   }
        self.apply_PBC_list    = {'single':self.apply_PBC_single   , 'multi':self.apply_PBC_multi   , 'mpi':self.apply_PBC_mpi   }
        self.apply_TFSF_list   = {'single':self.apply_TFSF_single  , 'multi':self.apply_TFSF_multi  , 'mpi':self.apply_TFSF_mpi  }
        self.apply_TFSF1D_list = {'single':self.apply_TFSF1D_single, 'multi':self.apply_TFSF1D_multi, 'mpi':self.apply_TFSF1D_mpi}
        self.update_e_list     = {'single':self.update_e_single    , 'multi':self.update_e_multi    , 'mpi':self.update_e_mpi    }
        self.update_h_list     = {'single':self.update_h_single    , 'multi':self.update_h_multi    , 'mpi':self.update_h_mpi    }
        self.updateE_list      = {'single':self.updateE_single     , 'multi':self.updateE_multi     , 'mpi':self.updateE_mpi     }
        self.updateH_list      = {'single':self.updateH_single     , 'multi':self.updateH_multi     , 'mpi':self.updateH_mpi     }
        self.init_list         = {'single':self.init_single        , 'multi':self.init_multi        , 'mpi':self.init_mpi        }
        self.set_structures_list = {'single':self.set_structures_single, 'multi':self.set_structures_multi, 'mpi':self.set_structures_mpi}
        self.apply_direct_source_list = {'single':self.apply_direct_source_single, 'multi':self.apply_direct_source_multi, 'mpi':self.apply_direct_source_mpi}
        self.apply_monochromatic_source_list = {'single':self.apply_monochromatic_source_single, 'multi':self.apply_monochromatic_source_multi, 'mpi':self.apply_monochromatic_source_mpi}
        self.apply_RFT_list = {'single':self.apply_RFT_single , 'multi':self.apply_RFT_multi , 'mpi':self.apply_RFT_mpi }
        self.get_info_list  = {'single':self.get_info_single, 'multi':self.get_info_multi, 'mpi':self.get_info_mpi}

    def init(self, opt='fields'):
        self.init_list[self.extension](opt)

    def get_info(self):
        self.get_info_list[self.extension]()

    def apply_PML(self, pml_apply, pml_thick={'x':(10,10),'y':(10,10),'z':(10,10)}, alpha0=0., kappa0=1., malpha=4, msigma=4, lnR0=-16.):
        self.is_pml = True
        self.pml_apply = pml_apply
        self.apply_PML_list[self.extension](pml_apply, pml_thick, alpha0, kappa0, malpha, msigma, lnR0)

    def apply_PBC(self, pbc_apply, klx=0., kly=0., klz=0.):
        self.is_pbc = True
        self.pbc_apply = pbc_apply
        self.bbc_coeff = (klx, kly, klz)
        self.apply_PBC_list[self.extension](pbc_apply, klx=klx, kly=kly, klz=klz)

    def apply_TFSF(self, region, boundary={'x':'+-', 'y':'+-', 'z':'+-'}, is_oblique=True):
        return self.apply_TFSF_list[self.extension](region, boundary, is_oblique)

    def apply_TFSF1D(self, field, region, rot_vec, pol_vec, boundary={'x':'+-', 'y':'+-', 'z':'+-'}, material=None):
        return self.apply_TFSF1D_list[self.extension](field, region, rot_vec, pol_vec, boundary, material)

    def apply_direct_source(self, field, region):
        src = self.apply_direct_source_list[self.extension](field, region)
        return src

    def apply_monochromatic_source(self, field, region, freq):
        src = self.apply_monochromatic_source_list[self.extension](field, region, freq)
        return src

    def apply_RFT(self, field, region, freq_domain):
        rft = self.apply_RFT_list[self.extension](field, region, freq_domain)
        return rft

    def set_structures(self, structures=None):
        self.set_structures_list[self.extension](structures)

    def update_e(self, wait=True):
        return self.update_e_list[self.extension](wait=wait)

    def update_h(self, wait=True):
        return self.update_h_list[self.extension](wait=wait)

    def updateE(self, wait=True):
        return self.updateE_list[self.extension](wait=wait)

    def updateH(self, wait=True):
        return self.updateH_list[self.extension](wait=wait)

def Basic_FDTD(mode, space_grid, dtype=np.float32, engine='nvidia_opencl', device_id=0, MPI_extension=False):
    if   mode == '1D':
        return Basic_FDTD_1d(mode, space_grid, dtype)
    elif mode in ['2DTE', '2DTM']:
        if isinstance(device_id, list) or isinstance(device_id, tuple):
            if len(device_id) >= 2:
                raise NotImplementedError, 'in 2-dimension FDTD, multi-devices environment is not implemented'
        if MPI_extension:
            raise NotImplementedError, 'in 2-dimension FDTD, MPI_extension environment is not implemented'
        return Basic_FDTD_2d(mode, space_grid, dtype, engine)
    elif mode == '3D':
        return Basic_FDTD_3d(mode, space_grid, dtype, engine, device_id, MPI_extension)
    else:
        raise TypeError, 'Invalid option \'device\': Plase check the input variable \'mode\''

def FDTD_world(mode, space_grid, dtype=np.float32, engine='nvidia_opencl', device_id=0, MPI_extension=False):
    return Basic_FDTD(mode, space_grid, dtype, device, device_id, is_subfdtd)
