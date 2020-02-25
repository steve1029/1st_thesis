# -*- coding:utf-8 -*-
import numpy as np

import exception
from util import *

class ndarray(object):
    def __init__(self, shape, dtype, mem_flag='rw'):
        if   dtype in complex_types:
            self.is_complex = True
            self.real = None
            self.imag = None
        elif dtype in float_types:
            self.is_complex = False
            self.real = self
            self.imag = None
        elif dtype in int_types:
            self.is_complex = False
            self.real = self
            self.imag = None
        else:
            raise TypeError, 'Attribute \'dtype\' has to be in %s' % str(int_types + float_types + complex_types)

        self.is_data = False
        self.is_buff = False
        self.shape = shape
        self.dtype = dtype
        self.size  = shape_to_size(self.shape)
        self.ndim  = len(self.shape)
        self.nbytes = self.size*dtype_to_bytes(self.dtype)
        self.mem_flag = mem_flag
        self.flags = {'data':self.is_data, 'complex':self.is_complex}
        self.use_buffer_mode = False

    def set_buffer(self, buff, engine):
        if 'opencl' in engine.name:
            self.engine= engine
            self.ctx   = engine.ctx
            self.queue = engine.queue
            self.prg   = engine.programs['get_set_data']
            self.ls    = engine.ls
            self.cl    = engine.cl
            self.mf    = self.cl.mem_flags

            mf = self.mf
            clmf = {'r':mf.READ_ONLY, 'w':mf.WRITE_ONLY, 'rw':mf.READ_WRITE}[self.mem_flag]

        elif 'cuda' in engine.name:
            self.engine= engine
            self.ctx   = engine.ctx
            self.queue = engine.queue
            self.prg   = engine.programs['get_set_data']
            self.cu    = engine.drv
            self.ls    = engine.ls
            self.get_data        = engine.kernels['get_data']
            self.set_data_svalue = engine.kernels['set_data_svalue']
            self.set_data_values = engine.kernels['set_data_values']

        elif 'cpu' in engine.name:
            self.engine = engine
            self.ctx    = engine.ctx
            self.queue  = engine.queue
            self.ls     = engine.ls
            self.prg    = None

        if not self.is_complex:
            if 'opencl' in engine.name:
                shape = np.array(self.shape, dtype=np.int32)
                digit = np.ones_like(shape, dtype=np.int32)
                for i, n in enumerate(shape):
                    digit[:i] *= n
                self.data = self.cl.Buffer(self.ctx, clmf, self.nbytes)
                if buff is not None:
                    self.cl.enqueue_write_buffer(self.queue, self.data, self.dtype(buff))
                self.digit_device = self.cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=digit)
                self.is_data   = True

                self.data_updated_flag = False
                self.buff_updated_flag = False
                self.buff = np.zeros(shape=self.shape, dtype=self.dtype)
                self.is_buff = True

            elif 'cuda' in engine.name:
                shape = np.array(self.shape, dtype=np.int32)
                digit = np.ones_like(shape, dtype=np.int32)
                for i, n in enumerate(shape):
                    digit[:i] *= n
                if buff is None:
                    self.data = engine.mem_alloc(self.nbytes)
                else:
                    self.data = engine.to_device(self.dtype(buff))
                self.digit_device = engine.to_device(digit)
                self.is_data   = True

                self.data_updated_flag = False
                self.buff_updated_flag = False
                self.buff = np.zeros(shape=self.shape, dtype=self.dtype)
                self.is_buff = True

            elif 'cpu' in engine.name:
                shape = np.array(self.shape, dtype=np.int32)
                digit = np.ones_like(shape, dtype=np.int32)
                for i, n in enumerate(shape):
                    digit[:i] *= n
                if buff is None:
                    self.data = np.zeros(self.shape, self.dtype)
                else:
                    self.data = buff.reshape(self.shape).astype(self.dtype)
                self.digit_device = digit
                self.is_data   = True

                self.data_updated_flag = False
                self.buff_updated_flag = False
                self.buff = None
                self.is_buff = False
        else:
            self.real = ndarray(self.shape, dtype=comp_to_real(self.dtype))
            self.imag = ndarray(self.shape, dtype=comp_to_real(self.dtype))
            slices = []
            for i in xrange(self.ndim):
                slices.append(slice(None,None,None))
            if buff is not None:
                buff_real = np.zeros_like(buff, dtype=comp_to_real(self.dtype))
                buff_imag = np.zeros_like(buff, dtype=comp_to_real(self.dtype))
                buff_real.__setitem__(slices, buff.real.__getitem__(slices))
                buff_imag.__setitem__(slices, buff.imag.__getitem__(slices))
            else:
                buff_real = None
                buff_imag = None
            self.real.set_buffer(buff_real, self.engine)
            self.imag.set_buffer(buff_imag, self.engine)

    def set_buffer_from_FDTD(self, buff, fdtd):
        self.fdtd   = fdtd
        fdtd.engine.allocated_mem_size += self.nbytes
        self.use_buffer_mode = fdtd.ndarray_use_buffer_mode
        fdtd.fields_list.append(self)
        if buff is not None:
            if buff.shape != self.shape:
                raise ValueError, 'Invalid buff shape'

        if   'opencl' in fdtd.engine.name:
            self.engine= fdtd.engine
            self.cl    = fdtd.engine.cl
            self.ctx   = fdtd.engine.ctx
            self.queue = fdtd.engine.queue
            self.prg   = fdtd.engine.programs['get_set_data']
            self.ls    = fdtd.engine.ls
            self.mf = self.cl.mem_flags
            mf = self.mf
            clmf = {'r':self.mf.READ_ONLY, 'w':self.mf.WRITE_ONLY, 'rw':mf.READ_WRITE}[self.mem_flag]

        elif   'cuda' in fdtd.engine.name:
            self.engine= fdtd.engine
            self.cu    = fdtd.engine.drv
            self.ctx   = fdtd.engine.ctx
            self.queue = fdtd.engine.queue
            self.prg   = fdtd.engine.programs['get_set_data']
            self.ls    = fdtd.engine.ls

            self.get_data        = fdtd.engine.kernels['get_data']
            self.set_data_svalue = fdtd.engine.kernels['set_data_svalue']
            self.set_data_values = fdtd.engine.kernels['set_data_values']

#            self.cst_strts = fdtd.engine.get_global(self.prg, 'cst_strts')
#            self.cst_steps = fdtd.engine.get_global(self.prg, 'cst_steps')
#            self.cst_d_dev = fdtd.engine.get_global(self.prg, 'cst_d_dev')
#            self.cst_d_buf = fdtd.engine.get_global(self.prg, 'cst_d_buf')

        elif    'cpu' in fdtd.engine.name:
            self.engine= fdtd.engine
            self.ctx   = fdtd.engine.ctx
            self.queue = fdtd.engine.queue
            self.ls    = fdtd.engine.ls

            fdtd.nbytes += self.nbytes

        if not self.is_complex:
            if   'opencl' in fdtd.engine.name:
                shape = np.array(self.shape, dtype=np.int32)
                digit = np.ones_like(shape, dtype=np.int32)
                for i, n in enumerate(shape):
                    digit[:i] *= n
                self.data = self.cl.Buffer(self.ctx, clmf, self.nbytes)
                if buff is not None:
                    self.cl.enqueue_write_buffer(self.queue, self.data, self.dtype(buff))
                self.digit_device = self.cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=digit)
                self.is_data = True

                self.data_updated_flag = False
                self.buff_updated_flag = False
                if buff is not None:
                    self.buff = buff
                    self.is_buff = True
                else:
                    #self.buff = np.zeros(shape=self.shape, dtype=self.dtype)
                    self.buff = None
                    self.is_buff = False

            elif 'cuda' in fdtd.engine.name:
                shape = np.array(self.shape, dtype=np.int32)
                digit = np.ones_like(shape, dtype=np.int32)
                for i, n in enumerate(shape):
                    digit[:i] *= n
                if buff is None:
                    self.data = fdtd.engine.mem_alloc(self.nbytes)
                else:
                    self.data = fdtd.engine.to_device(self.dtype(buff))
                self.digit_device = fdtd.engine.to_device(digit)
                self.is_data = True

                self.data_updated_flag = False
                self.buff_updated_flag = False
                if buff is not None:
                    self.buff = buff
                    self.is_buff = True
                else:
                    #self.buff = np.zeros(shape=self.shape, dtype=self.dtype)
                    self.buff = None
                    self.is_buff = False

            elif  'cpu' in fdtd.engine.name:
                shape = np.array(self.shape, dtype=np.int32)
                digit = np.ones_like(shape, dtype=np.int32)
                for i, n in enumerate(shape):
                    digit[:i] *= n
                if buff is None:
                    self.data = np.zeros(self.shape, self.dtype)
                else:
                    self.data = buff.reshape(self.shape).astype(self.dtype)
                self.digit_device = digit
                self.is_data = True

                self.data_updated_flag = False
                self.buff_updated_flag = False
                self.buff = None
                self.is_buff = False
        else:
            self.real = ndarray(self.shape, dtype=comp_to_real(self.dtype), mem_flag=self.mem_flag)
            self.imag = ndarray(self.shape, dtype=comp_to_real(self.dtype), mem_flag=self.mem_flag)
            self.data_updated_flag = False
            self.buff_updated_flag = False
            slices = []
            for i in xrange(self.ndim):
                slices.append(slice(None,None,None))
            if buff is not None:
                buff_real = np.zeros_like(buff, dtype=comp_to_real(self.dtype))
                buff_imag = np.zeros_like(buff, dtype=comp_to_real(self.dtype))
                buff_real.__setitem__(slices, buff.real.__getitem__(slices))
                buff_imag.__setitem__(slices, buff.imag.__getitem__(slices))
            else:
                buff_real = None
                buff_imag = None
            self.real.set_buffer(buff_real, self.engine)
            self.imag.set_buffer(buff_imag, self.engine)

    def release_data(self):
        if self.is_complex:
            if   'opencl' in self.fdtd.engine.name:
                self.real.data.release()
                self.imag.data.release()
                if self.real.is_buff:
                    del self.real.buff
                if self.imag.is_buff:
                    del self.imag.buff
            elif   'cuda' in self.fdtd.engine.name:
                self.fdtd.engine.enqueue(self.real.data.free, [])
                self.fdtd.engine.enqueue(self.imag.data.free, [])
                if self.real.is_buff:
                    del self.real.buff
                if self.imag.is_buff:
                    del self.imag.buff
            elif   'cpu' in self.fdtd.engine.name:
                del self.real.data
                del self.imag.data
        else:
            if   'opencl' in self.fdtd.engine.name:
                self.data.release()
                if self.is_buff:
                    del self.buff
            elif   'cuda' in self.fdtd.engine.name:
                self.fdtd.engine.enqueue(self.data.free, [])
                if self.is_buff:
                    del self.buff
            elif   'cpu' in self.fdtd.engine.name:
                del self.data
        self.fdtd.engine.allocated_mem_size -= self.nbytes

    def __del__(self):
        self.release_data()

    def to_indices(self, key):
        strts = np.array([0 for i in xrange(self.ndim)], dtype=np.int32)
        stops = np.array([0 for i in xrange(self.ndim)], dtype=np.int32)
        steps = np.array([1 for i in xrange(self.ndim)], dtype=np.int32)
        n_h   = np.array([1 for i in xrange(self.ndim)], dtype=np.int32)

        if self.ndim == 1:
            n = self.shape[0]
            if   isinstance(key, int_types):
                if key < 0:
                    strts[0] = key + n
                    stops[0] = key + n
                else:
                    strts[0] = key
                    stops[0] = key

            elif isinstance(key, slice):
                strt = key.start
                stop = key.stop
                step = key.step

                if   step == None: step  =   1
                elif step == 0:
                    raise ValueError, 'slice.step must not be 0'

                if   strt == None:
                    if step >   0: strt  =   0
                    else         : strt  = n-1
                elif strt  <    0: strt += n

                if   stop == None:
                    if step >   0: stop  = n-1
                    else         : stop  =   0
                elif stop   >   n: stop  = n-1
                elif stop   <   0: stop += n-1
                elif stop   >   0: stop -=   1

                strts[0] = strt
                stops[0] = stop
                steps[0] = step
                n_test   =(stop - strt)/abs(step) + 1
                if n_test <= 0:
                    raise IndexError, 'Invalid array index'
                n_h  [0] = n_test
            else:
                raise TypeError, 'Key of Fields\' class instance must be the type of \'int\' or \'slice\''

        else:
            for i, n in enumerate(self.shape):
                if   isinstance(key[i], int_types):
                    if key[i] < 0:
                        strts[i] = key[i] + n
                        stops[i] = key[i] + n
                    else:
                        strts[i] = key[i]
                        stops[i] = key[i]

                elif isinstance(key[i], slice):
                    strt = key[i].start
                    stop = key[i].stop
                    step = key[i].step

                    if   step == None: step  =   1
                    elif step == 0:
                        raise ValueError, 'slice.step must not be 0'

                    if   strt == None:
                        if step >   0: strt  =   0
                        else         : strt  = n-1
                    elif strt  <    0: strt +=   n

                    if   stop == None:
                        if step >   0: stop  = n-1
                        else         : stop  =   0
                    elif stop   >   n: stop  = n-1
                    elif stop   <   0: stop += n-1
                    elif stop   >   0: stop -= 1

                    strts[i] = strt
                    stops[i] = stop
                    steps[i] = step
                    n_test   =(stop - strt)/abs(step) + 1
                    if n_test <= 0:
                        raise IndexError, 'Invalid array index'
                    n_h  [i] = n_test
                else:
                    raise TypeError, 'Key of Fields\' class instance must be the type of \'int\' or \'slice\''
        return strts, stops, steps, n_h

    def __getitem__(self, key):
        if not self.is_complex:
            if 'cpu' in self.engine.name:
                return self.data.__getitem__(key)
            if self.use_buffer_mode:
                if self.data_updated_flag:
                    if 'opencl' in self.engine.name:
                        self.cl.enqueue_read_buffer(self.queue, self.data, self.buff).wait()
                    elif 'cuda' in self.engine.name:
                        self.engine.enqueue(self.cu.memcpy_dtoh, [self.buff, self.data]).wait()
                self.buff_updated_flag = False
                self.data_updated_flag = False
                return self.buff.__getitem__(key)
            strts, stops, steps, n_h = self.to_indices(key)
            digit = np.ones_like(n_h)
            size_h = 1
            for i, n in enumerate(n_h):
                size_h    *= n
                digit[:i] *= n
            nbytes_h = int(size_h*dtype_to_bytes(self.dtype))
            digit_device = self.digit_device

            if   'opencl' in self.engine.name:
                gs = cl_global_size(size_h, self.ls)
                mf = self.mf
                cl = self.cl
#                strts_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=strts)
#                steps_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=steps)
#                digit_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=digit)
                strts_buffer = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=strts)
                steps_buffer = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=steps)
                digit_buffer = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=digit)
                data_buffer  = cl.Buffer(self.ctx, mf.WRITE_ONLY, nbytes_h)
                args = [self.queue, (gs,), (self.ls,), \
                        np.int32(self.ndim), np.int32(size_h), \
                        strts_buffer, \
                        steps_buffer, \
                        digit_device, \
                        digit_buffer, \
                        self.data, \
                        data_buffer]
                self.prg.get_data(*args).wait()
                data_host = np.zeros(size_h, dtype=self.dtype)
                cl.enqueue_read_buffer(self.queue, data_buffer, data_host).wait()

#                strts_buffer.release()
#                steps_buffer.release()
#                digit_buffer.release()
#                data_buffer.release()
#                del strts_buffer, steps_buffer, digit_buffer, data_buffer

            elif 'cuda' in self.engine.name:
                gs = size_h
                cu = self.engine
                strts_buffer = cu.to_device(strts)
                steps_buffer = cu.to_device(steps)
                digit_buffer = cu.to_device(digit)
                data_buffer  = cu.mem_alloc(nbytes_h)

#                cu.enqueue(cu.drv.memcpy_dtod, [self.cst_strts, strts_buffer, self.ndim*4])
#                cu.enqueue(cu.drv.memcpy_dtod, [self.cst_steps, steps_buffer, self.ndim*4])
#                cu.enqueue(cu.drv.memcpy_dtod, [self.cst_d_dev, digit_device, self.ndim*4])
#                cu.enqueue(cu.drv.memcpy_dtod, [self.cst_d_buf, digit_buffer, self.ndim*4])

#                d_dev = cu.from_device(digit_device, (self.ndim,), np.int32)
#                cu.enqueue(cu.drv.memcpy_htod, [self.cst_strts, strts])
#                cu.enqueue(cu.drv.memcpy_htod, [self.cst_steps, steps])
#                cu.enqueue(cu.drv.memcpy_htod, [self.cst_d_dev, d_dev])
#                cu.enqueue(cu.drv.memcpy_htod, [self.cst_d_buf, digit])

                args = [self.queue, (gs,), (self.ls,), \
                        np.int32(self.ndim), np.int32(size_h), \
                        strts_buffer, \
                        steps_buffer, \
                        digit_device, \
                        digit_buffer, \
                        self.data, \
                        data_buffer]
                cu.enqueue_kernel(self.get_data, args, True)

                data_host = cu.from_device(data_buffer, (size_h,), self.dtype)

#                cu.enqueue(strts_buffer.free, [])
#                cu.enqueue(steps_buffer.free, [])
#                cu.enqueue(digit_buffer.free, [])
#                cu.enqueue( data_buffer.free, [])
#                del strts_buffer, steps_buffer, digit_buffer, data_buffer

            if self.ndim == 1:
                if   isinstance(key, int_types):
                    return data_host[0]
                elif isinstance(key, slice):
                    return data_host
            else:
                shape = []
                for i in xrange(self.ndim):
                    if   isinstance(key[i], int_types):
                        pass
                    elif isinstance(key[i], slice):
                        shape.append(n_h[i])
                if shape == []:
                    return data_host[0]
                else:
                    return data_host.reshape(shape)
        else:
            return self.real.__getitem__(key) + 1.j*self.imag.__getitem__(key)

    def setitem_operation_type(self, key, values, operation_type):
        opt  = operation_type
        if not self.is_complex:
            if 'cpu' in self.engine.name:
                if   operation_type is 0:
                    self.data.__setitem__(key, values)
                elif operation_type is 1:
                    self.data.__add__(key, values)
                elif operation_type is 2:
                    self.data.__sub__(key, values)
                elif operation_type is 3:
                    self.data.__mul__(key, values)
                elif operation_type is 4:
                    self.data.__div__(key, values)
                return None
            strts, stops, steps, n_h = self.to_indices(key)
            digit = np.ones_like(n_h)
            size_h = 1
            for i, n in enumerate(n_h):
                size_h    *= n
                digit[:i] *= n
            nbytes_h = int(size_h*dtype_to_bytes(self.dtype))

            if self.use_buffer_mode:
                if self.data_updated_flag:
                    if 'opencl' in self.engine.name:
                        self.cl.enqueue_read_buffer(self.queue, self.data, self.buff).wait()
                    elif 'cuda' in self.engine.name:
                        self.engine.enqueue(self.cu.memcpy_dtoh, [self.buff, self.data]).wait()
                    self.data_updated_flag = False
                self.buff_updated_flag = True
                if   operation_type is 0:
                    self.buff.__setitem__(key, values)
                elif operation_type is 1:
                    self.buff.__add__(key, values)
                elif operation_type is 2:
                    self.buff.__sub__(key, values)
                elif operation_type is 3:
                    self.buff.__mul__(key, values)
                elif operation_type is 4:
                    self.buff.__div__(key, values)
                return None

            if   'opencl' in self.engine.name:
                gs = cl_global_size(size_h, self.ls)
                cl = self.cl
                mf = self.mf
                digit_device = self.digit_device
#                strts_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=strts)
#                steps_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=steps)
#                digit_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=digit)
                strts_buffer = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=strts)
                steps_buffer = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=steps)
                digit_buffer = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=digit)
                data_buffer  = cl.Buffer(self.ctx, mf.WRITE_ONLY, nbytes_h)

                args = [self.queue, (gs,), (self.ls,), \
                        np.int32(self.ndim), np.int32(size_h), np.int32(opt), \
                        strts_buffer, \
                        steps_buffer, \
                        digit_device, \
                        digit_buffer, \
                        self.data]

                if   isinstance(values, float_types + int_types):
                    args += [self.dtype(values)]
                    self.prg.set_data_svalue(*args).wait()
                elif isinstance(values, np.ndarray):
                    values_r = self.dtype(values)
                    values_s = values_r.reshape(values_r.size)
                    values_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=values_s)
                    args += [values_buffer]
                    self.prg.set_data_values(*args).wait()
                else:
                    raise TypeError, 'Input Value has wrong type'

            elif 'cuda' in self.engine.name:
                gs = size_h
                cu = self.engine
                digit_device = self.digit_device
                strts_buffer = cu.to_device(strts)
                steps_buffer = cu.to_device(steps)
                digit_buffer = cu.to_device(digit)
                data_buffer  = cu.mem_alloc(nbytes_h)

#                cu.enqueue(cu.drv.memcpy_dtod, [self.cst_strts, strts_buffer, self.ndim*4]).wait()
#                cu.enqueue(cu.drv.memcpy_dtod, [self.cst_steps, steps_buffer, self.ndim*4]).wait()
#                cu.enqueue(cu.drv.memcpy_dtod, [self.cst_d_dev, digit_device, self.ndim*4]).wait()
#                cu.enqueue(cu.drv.memcpy_dtod, [self.cst_d_buf, digit_buffer, self.ndim*4]).wait()

#                d_dev = cu.from_device(digit_device, (self.ndim,), np.int32)
#                cu.enqueue(cu.drv.memcpy_htod, [self.cst_strts, strts])
#                cu.enqueue(cu.drv.memcpy_htod, [self.cst_steps, steps])
#                cu.enqueue(cu.drv.memcpy_htod, [self.cst_d_dev, d_dev])
#                cu.enqueue(cu.drv.memcpy_htod, [self.cst_d_buf, digit])

                args = [self.queue, (gs,), (self.ls,), \
                        np.int32(self.ndim), np.int32(size_h), np.int32(opt), \
                        strts_buffer, \
                        steps_buffer, \
                        digit_device, \
                        digit_buffer, \
                        self.data]

                if   isinstance(values, float_types + int_types):
                    args += [self.dtype(values)]
                    cu.enqueue_kernel(self.set_data_svalue, args, True)
                elif isinstance(values, np.ndarray):
                    values_r = self.dtype(values)
                    values_s = values_r.reshape(values_r.size)
                    values_buffer = cu.to_device(values_s)
                    args += [values_buffer]
                    cu.enqueue_kernel(self.set_data_values, args, True)
                else:
                    raise TypeError, 'Input Value has wrong type'

        else:
            if isinstance(values, np.ndarray):
                '''
                values_real = np.zeros_like(values, dtype=comp_to_real(self.dtype))
                values_imag = np.zeros_like(values, dtype=comp_to_real(self.dtype))
                slices = []
                for i in xrange(values.ndim):
                    slices.append(slice(None,None,None))
                values_real.__setitem__(slices, values.real.__getitem__(slices))
                values_imag.__setitem__(slices, values.imag.__getitem__(slices))
                self.real.setitem_operation_type(key, values_real, operation_type)
                self.imag.setitem_operation_type(key, values_imag, operation_type)
                '''
                self.real.setitem_operation_type(key, np.ascontiguousarray(values.real), operation_type)
                self.imag.setitem_operation_type(key, np.ascontiguousarray(values.imag), operation_type)
            else:
                self.real.setitem_operation_type(key, values.real, operation_type)
                self.imag.setitem_operation_type(key, values.imag, operation_type)

    def __setitem__(self, key, value):
        if 'cpu' in self.engine.name:
            self.real.data.__setitem__(key, np.real(value))
            if self.is_complex:
                self.imag.data.__setitem__(key, np.imag(value))
        else:
            self.setitem_operation_type(key, value, 0)

    def set_fields(self, key, value, opt):
        opt_to_int = {None:0, '=':0, '+':1, '-':2, '*':3, '/':4}
        opt_int = opt_to_int[opt]
        self.setitem_operation_type(key, value, opt_int)

class Fields(ndarray):
    def __init__(self, fdtd, shape, dtype, mem_flag='rw', init_value=None, name=None):
        ndarray.__init__(self, shape, dtype=dtype, mem_flag=mem_flag)
        ndarray.set_buffer_from_FDTD(self, init_value, fdtd)
        self.name = name
        if self.ndim >= 1:
            self.nx_d = self.shape[0]
        if self.ndim >= 2:
            self.ny_d = self.shape[1]
        if self.ndim >= 3:
            self.nz_d = self.shape[2]

    def get(self, key):
        self.use_buffer_mode = True
        got = self.__getitem__(key)
        self.use_buffer_mode = self.fdtd.ndarray_use_buffer_mode
        return got

    def set(self, key, value):
        self.use_buffer_mode = True
        self.__setitem__(key, value)
        self.use_buffer_mode = self.fdtd.ndarray_use_buffer_mode

    def buff_to_data(self, wait=True):
        evts = []
        if 'opencl' in self.engine.name:
            evt = self.cl.enqueue_write_buffer(self.queue, self.real.data, self.real.buff)
        elif 'cuda' in self.engine.name:
            evt = self.engine.enqueue(self.cu.memcpy_htod, [self.real.data, self.real.buff])
        elif  'cpu' in self.engine.name:
            evt = FakeEvent()
        evts.append(evt)
        self.real.buff_updated_flag = False
        if self.is_complex:
            if 'opencl' in self.engine.name:
                evt = self.cl.enqueue_write_buffer(self.queue, self.imag.data, self.imag.buff)
            elif 'cuda' in self.engine.name:
                evt = self.engine.enqueue(self.cu.memcpy_htod, [self.imag.data, self.imag.buff])
            elif  'cpu' in self.engine.name:
                evt = FakeEvent()
            evts.append(evt)
            self.imag.buff_updated_flag = False
        if wait: wait_for_events(self.fdtd, evts)
        self.buff_updated_flag = False
        return evts

    def data_to_buff(self, wait=True):
        evts = []
        if 'opencl' in self.engine.name:
            evt = self.cl.enqueue_read_buffer(self.queue, self.real.data, self.real.buff)
        elif 'cuda' in self.engine.name:
            evt = self.engine.enqueue(self.cu.memcpy_dtoh, [self.real.buff, self.real.data])
        elif  'cpu' in self.engine.name:
            evt = FakeEvent()
        if wait: evt.wait()
        evts.append(evt)
        self.real.data_updated_flag = False
        if self.is_complex:
            if 'opencl' in self.engine.name:
                evt = self.cl.enqueue_read_buffer(self.queue, self.imag.data, self.imag.buff)
            elif 'cuda' in self.engine.name:
                evt = self.engine.enqueue(self.cu.memcpy_dtoh, [self.imag.buff, self.imag.data])
            elif  'cpu' in self.engine.name:
                evt = FakeEvent()
            evts.append(evt)
            self.imag.buff_updated_flag = False
        if wait: wait_for_events(self.fdtd, evts)
        self.data_updated_flag = False
        return evts

    def __del__(self):
        self.fdtd.fields_list.remove(self)
        ndarray.__del__(self)

    def __getitem__(self, key):
        return ndarray.__getitem__(self, key)

    def __setitem__(self, key, value):
        ndarray.__setitem__(self, key, value)
        '''
        if   '2D' in self.fdtd.mode:
            ce1, ch1 = 1., 1.
            ce2, ch2 = self.fdtd.dt, self.fdtd.dt
            slc = slice(None, None, None)
            if self.name == 'ce1x':
                ndarray.setitem_operation_type(self, (slc, 0), ce1, 0)
            if self.name == 'ce1y':
                ndarray.setitem_operation_type(self, (0, slc), ce1, 0)
            if self.name == 'ce1z':
                ndarray.setitem_operation_type(self, (0, slc), ce1, 0)
                ndarray.setitem_operation_type(self, (slc, 0), ce1, 0)
            if self.name == 'ce2x':
                ndarray.setitem_operation_type(self, (slc, 0), ce2, 0)
            if self.name == 'ce2y':
                ndarray.setitem_operation_type(self, (0, slc), ce2, 0)
            if self.name == 'ce2z':
                ndarray.setitem_operation_type(self, (0, slc), ce2, 0)
                ndarray.setitem_operation_type(self, (slc, 0), ce2, 0)

            if self.name == 'ch1x':
                ndarray.setitem_operation_type(self, (slc, -1), ch1, 0)
            if self.name == 'ch1y':
                ndarray.setitem_operation_type(self, (-1, slc), ch1, 0)
            if self.name == 'ch1z':
                ndarray.setitem_operation_type(self, (-1, slc), ch1, 0)
                ndarray.setitem_operation_type(self, (slc, -1), ch1, 0)
            if self.name == 'ch2x':
                ndarray.setitem_operation_type(self, (slc, -1), ch2, 0)
            if self.name == 'ch2y':
                ndarray.setitem_operation_type(self, (-1, slc), ch2, 0)
            if self.name == 'ch2z':
                ndarray.setitem_operation_type(self, (-1, slc), ch2, 0)
                ndarray.setitem_operation_type(self, (slc, -1), ch2, 0)

        elif '3D' in self.fdtd.mode:
            ce1, ch1 = 1., 1.
            ce2, ch2 = self.fdtd.dt, self.fdtd.dt
            slc = slice(None, None, None)
            if self.name == 'ce1x':
                ndarray.setitem_operation_type(self, (slc, 0, slc), ce1, 0)
                ndarray.setitem_operation_type(self, (slc, slc, 0), ce1, 0)
            if self.name == 'ce1y':
                ndarray.setitem_operation_type(self, (slc, slc, 0), ce1, 0)
                ndarray.setitem_operation_type(self, (0, slc, slc), ce1, 0)
            if self.name == 'ce1z':
                ndarray.setitem_operation_type(self, (0, slc, slc), ce1, 0)
                ndarray.setitem_operation_type(self, (slc, 0, slc), ce1, 0)
            if self.name == 'ce2x':
                ndarray.setitem_operation_type(self, (slc, 0, slc), ce2, 0)
                ndarray.setitem_operation_type(self, (slc, slc, 0), ce2, 0)
            if self.name == 'ce2y':
                ndarray.setitem_operation_type(self, (slc, slc, 0), ce2, 0)
                ndarray.setitem_operation_type(self, (0, slc, slc), ce2, 0)
            if self.name == 'ce2z':
                ndarray.setitem_operation_type(self, (0, slc, slc), ce2, 0)
                ndarray.setitem_operation_type(self, (slc, 0, slc), ce2, 0)

            if self.name == 'ch1x':
                ndarray.setitem_operation_type(self, (slc, -1, slc), ch1, 0)
                ndarray.setitem_operation_type(self, (slc, slc, -1), ch1, 0)
            if self.name == 'ch1y':
                ndarray.setitem_operation_type(self, (slc, slc, -1), ch1, 0)
                ndarray.setitem_operation_type(self, (-1, slc, slc), ch1, 0)
            if self.name == 'ch1z':
                ndarray.setitem_operation_type(self, (-1, slc, slc), ch1, 0)
                ndarray.setitem_operation_type(self, (slc, -1, slc), ch1, 0)
            if self.name == 'ch2x':
                ndarray.setitem_operation_type(self, (slc, -1, slc), ch2, 0)
                ndarray.setitem_operation_type(self, (slc, slc, -1), ch2, 0)
            if self.name == 'ch2y':
                ndarray.setitem_operation_type(self, (slc, slc, -1), ch2, 0)
                ndarray.setitem_operation_type(self, (-1, slc, slc), ch2, 0)
            if self.name == 'ch2z':
                ndarray.setitem_operation_type(self, (-1, slc, slc), ch2, 0)
                ndarray.setitem_operation_type(self, (slc, -1 ,slc), ch2, 0)
        '''

class Fields_multi_devices(ndarray):
    def __init__(self, fdtd_multidev, shape, dtype, mem_flag='rw', name=None):
        self.dtype = dtype
        self.ndim  = len(shape)
        self.fdtd   = fdtd_multidev
        self.name   = name
        self.shape  = shape
        self.ndata  = len(self.fdtd.fdtd_group)
        self.mem_flag = mem_flag
        if self.dtype in complex_types:
            self.is_complex = True
        else:
            self.is_complex = False
        if not self.is_complex:
            self.bind_buffer_from_FDTD('real')
        else:
            self.bind_buffer_from_FDTD('comp')

    def bind_buffer_from_FDTD(self, part):
        if   part == 'real':
            self.real = self
            self.imag = None
            self.datas = []
            for i in xrange(self.ndata):
                self.datas.append(self.fdtd.fdtd_group[i].__dict__[self.name].real)
            self.is_datas = True
        elif part == 'imag':
            self.real = self
            self.imag = None
            self.datas = []
            for i in xrange(self.ndata):
                self.datas.append(self.fdtd.fdtd_group[i].__dict__[self.name].imag)
            self.is_datas = True
        elif part == 'comp':
            self.real = Fields_multi_devices(self.fdtd, self.shape, dtype=comp_to_real(self.dtype), mem_flag=self.mem_flag, name=self.name)
            self.imag = Fields_multi_devices(self.fdtd, self.shape, dtype=comp_to_real(self.dtype), mem_flag=self.mem_flag, name=self.name)
            self.real.bind_buffer_from_FDTD('real')
            self.imag.bind_buffer_from_FDTD('imag')
            self.is_datas = False

    def __getitem__(self, key):
        if not self.is_complex:
            strts, stops, steps, n_h = ndarray.to_indices(self, key)
            stops[0] = strts[0] + steps[0]*(n_h[0]-1)
            x_strt = strts[0]
            x_stop = stops[0]
            x_step = steps[0]

            shape = []
            for i in xrange(self.ndim):
                if   isinstance(key[i], int_types):
                    pass
                elif isinstance(key[i], slice):
                    shape.append(n_h[i])
            if shape == []:
                value = 0.
            else:
                value = np.zeros(tuple(shape), dtype=self.dtype)

            for i in xrange(self.ndata):
                if   x_strt == 0:
                    i_strt = 0
                elif x_strt > self.fdtd.x_strt_group[i] and x_strt <= self.fdtd.x_stop_group[i]:
                    i_strt = i
                if   x_stop == 0:
                    i_stop = 0
                elif x_stop > self.fdtd.x_strt_group[i] and x_stop <= self.fdtd.x_stop_group[i]:
                    i_stop = i

            for i in xrange(i_strt, i_stop+1):
                hostkey = []
                newkey = [None for ii in xrange(self.ndim)]
                for k in key:
                    if   isinstance(k,     slice): hostkey.append(slice(None,None,None))
                if hostkey == []:
                    if shape == []:
                        hostkey = [0]
                    else:
                        hostkey = [0 for ii in xrange(len(shape))]
                x_strt_d = x_strt - self.fdtd.x_strt_group[i]
                x_stop_d = x_stop - self.fdtd.x_strt_group[i]
                x_strt_h = self.fdtd.x_strt_group[i] - x_strt
                x_stop_h = self.fdtd.x_stop_group[i] - x_strt
                for k in xrange(self.ndim):
                    if k == 0:
                        if   i == i_strt and i == i_stop:
                            if   isinstance(key[0], int_types): newkey[0] =   int(x_strt_d)
                            elif isinstance(key[0],     slice): newkey[0] = slice(x_strt_d, x_stop_d+1, x_step)
                            hostkey[0] = slice(None,None,None)
                        elif i == i_strt:
                            if   isinstance(key[0], int_types): newkey[0] =   int(x_strt_d)
                            elif isinstance(key[0],     slice): newkey[0] = slice(x_strt_d, None, x_step)
                            hostkey[0] = slice(None,x_stop_h+1,None)
                        elif i != i_stop:
                            if   isinstance(key[0], int_types): newkey[0] =   int(       0)
                            elif isinstance(key[0],     slice): newkey[0] = slice(       0, None, x_step)
                            hostkey[0] = slice(x_strt_h,x_stop_h+1,None)
                        else:
                            if   isinstance(key[0], int_types): newkey[0] =   int(       0)
                            elif isinstance(key[0],     slice): newkey[0] = slice(       0, x_stop_d+1, x_step)
                            hostkey[0] = slice(x_strt_h,None,None)
                    else:
                        if   isinstance(key[k], int_types): newkey[k] =   int(strts[k])
                        elif isinstance(key[k],     slice): newkey[k] = slice(strts[k], stops[k]+1, steps[k])
                if shape == []:
                    value = self.datas[i][newkey]
                else:
                    value[hostkey] = self.datas[i][newkey]
            return value
        else:
            return self.real.__getitem__(key) + 1.j*self.imag.__getitem__(key)

    def setitem_operation_type(self, key, value, operation_type):
        if not self.is_complex:
            strts, stops, steps, n_h = ndarray.to_indices(self, key)
            stops[0] = strts[0] + steps[0]*(n_h[0]-1)

            x_strt = strts[0]
            x_stop = stops[0]
            x_step = steps[0]

            if x_strt == x_stop:
                divided = False
            else:
                divided = True

            for i in xrange(self.ndata):
                if   x_strt == 0:
                    i_strt = 0
                elif x_strt >  self.fdtd.x_strt_group[i] and x_strt <= self.fdtd.x_stop_group[i]:
                    i_strt = i
                if   x_stop == 0:
                    i_stop = 0
                if   x_stop >= self.fdtd.x_strt_group[i] and x_stop <= self.fdtd.x_stop_group[i]:
                    i_stop  = i

            shape = []
            for i in xrange(self.ndim):
                if   isinstance(key[i], int_types):
                    pass
                elif isinstance(key[i], slice):
                    shape.append(n_h[i])

            for i in xrange(i_strt, i_stop+1):
                hostkey = []
                newkey = [None for ii in xrange(self.ndim)]
                for k in key:
                    if   isinstance(k,     slice): hostkey.append(slice(None,None,None))
                if hostkey == []:
                    if shape == []:
                        hostkey = [0]
                    else:
                        hostkey = [0 for ii in xrange(len(shape))]
                x_strt_d = x_strt - self.fdtd.x_strt_group[i]
                x_stop_d = x_stop - self.fdtd.x_strt_group[i]
                x_strt_h = self.fdtd.x_strt_group[i] - x_strt
                x_stop_h = self.fdtd.x_stop_group[i] - x_strt
                for k in xrange(self.ndim):
                    if k == 0:
                        if   i == i_strt and i == i_stop:
                            if   isinstance(key[0], int_types): newkey[0] =   int(x_strt_d)
                            elif isinstance(key[0],     slice): newkey[0] = slice(x_strt_d, x_stop_d+1, x_step)
                            if                         divided:hostkey[0] = slice(None,None,None)
                        elif i == i_strt:
                            if   isinstance(key[0], int_types): newkey[0] =   int(x_strt_d)
                            elif isinstance(key[0],     slice): newkey[0] = slice(x_strt_d, None, x_step)
                            if                         divided:hostkey[0] = slice(None,x_stop_h+1,None)
                        elif i != i_stop:
                            if   isinstance(key[0], int_types): newkey[0] =   int(       0)
                            elif isinstance(key[0],     slice): newkey[0] = slice(       0, None, x_step)
                            if                         divided:hostkey[0] = slice(x_strt_h,x_stop_h+1,None)
                        else:
                            if   isinstance(key[0], int_types): newkey[0] =   int(       0)
                            elif isinstance(key[0],     slice): newkey[0] = slice(       0, x_stop_d+1, x_step)
                            if                         divided:hostkey[0] = slice(x_strt_h,None,None)
                    else:
                        if   isinstance(key[k], int_types): newkey[k] = int(strts[k])
                        elif isinstance(key[k],     slice): newkey[k] = slice(strts[k], stops[k]+1, steps[k])
                if isinstance(value, np.ndarray):
                    self.datas[i].setitem_operation_type(newkey, value.__getitem__(hostkey), operation_type)
                else:
                    self.datas[i].setitem_operation_type(newkey, value                     , operation_type)
        else:
            values_real = np.zeros_like(values, dtype=comp_to_real(self.dtype))
            values_imag = np.zeros_like(values, dtype=comp_to_real(self.dtype))
            slices = []
            for i in xrange(self.ndim):
                slices.append(slice(None,None,None))
            values_real.__setitem__(slices, values.real.__getitem__(slices))
            values_imag.__setitem__(slices, values.imag.__getitem__(slices))
            self.real.setitem_operation_type(key, values_real, operation_type)
            self.imag.setitem_operation_type(key, values_imag, operation_type)

    def __setitem__(self, key, value):
        self.setitem_operation_type(key, value, 0)

    def get(self, key):
        if not self.is_complex:
            for data in self.datas:
                data.use_buffer_mode = True
            got = self.__getitem__(key)
            for data in self.datas:
                data.use_buffer_mode = data.fdtd.ndarray_use_buffer_mode
        else:
            got = self.real.get(key) + 1.j*self.imag.get(key)
        return got

    def set(self, key, value):
        if not self.is_complex:
            for data in self.datas:
                data.use_buffer_mode = True
            self.__setitem__(key, value)
            for data in self.datas:
                data.use_buffer_mode = data.fdtd.ndarray_use_buffer_mode
        else:
            value_real = np.ascontiguousarray(np.real(value))
            value_imag = np.ascontiguousarray(np.imag(value))
            self.real.set(key, value_real)
            self.imag.set(key, value_imag)

    def set_fields(self, key, value, opt):
        opt_to_int = {None:0, '=':0, '+':1, '-':2, '*':3, '/':4}
        opt_int = opt_to_int[opt]
        self.setitem_operation_type(key, value, opt_int)

class Fields_MPI(ndarray):
    def __init__(self, fdtd_mpi, shape, dtype, mem_flag='rw', name=None):
        self.fdtd   = fdtd_mpi
        self.shape  = shape
        self.ndim   = len(shape)
        self.dtype  = dtype
        self.rank   = self.fdtd.rank
        self.name   = name
        self.shape  = shape
        self.mem_flag = mem_flag
        if self.dtype in complex_types:
            self.is_complex = True
        else:
            self.is_complex = False
        if not self.is_complex:
            self.bind_buffer_from_FDTD('real')
        else:
            self.bind_buffer_from_FDTD('comp')
        self.access = True

    def bind_buffer_from_FDTD(self, part):
        if   part == 'real':
            self.real = self
            self.imag = None
            if self.fdtd.worker:
                self.data = self.fdtd.sub_fdtd.__dict__[self.name].real
                self.is_data = True
            else:
                self.data = None
                self.is_data = False
        elif part == 'imag':
            self.real = self
            self.imag = None
            if self.fdtd.worker:
                self.data = self.fdtd.sub_fdtd.__dict__[self.name].imag
                self.is_data = True
            else:
                self.data = None
                self.is_data = False
        elif part == 'comp':
            self.real = Fields_MPI(self.fdtd, self.shape, dtype=comp_to_real(self.dtype), mem_flag=self.mem_flag, name=self.name)
            self.imag = Fields_MPI(self.fdtd, self.shape, dtype=comp_to_real(self.dtype), mem_flag=self.mem_flag, name=self.name)
            self.real.bind_buffer_from_FDTD('real')
            self.imag.bind_buffer_from_FDTD('imag')
            self.is_data = False

    def __getitem__(self, key):
        if not self.is_complex:
            strts, stops, steps, n_h = ndarray.to_indices(self, key)
            stops[0] = strts[0] + steps[0]*(n_h[0]-1)
            x_strt = strts[0]
            x_stop = stops[0]
            x_step = steps[0]

            for i in xrange(self.fdtd.size):
                if   x_strt == 0:
                    i_strt = 0
                elif x_strt > self.fdtd.x_strt_group[i] and x_strt <= self.fdtd.x_stop_group[i]:
                    i_strt = i
                if   x_stop > self.fdtd.x_strt_group[i] and x_stop <= self.fdtd.x_stop_group[i]:
                    i_stop = i

            shape = []
            for i in xrange(self.ndim):
                if   isinstance(key[i], int_types):
                    pass
                elif isinstance(key[i], slice):
                    shape.append(n_h[i])
            if shape == []:
                value = 0.
            else:
                value = np.zeros(tuple(shape), dtype=self.dtype)

            hostkey = []
            newkey = [None for ii in xrange(self.ndim)]
            for k in key:
                if   isinstance(k,     slice): hostkey.append(slice(None,None,None))
            if hostkey == []:
                if shape == []:
                    hostkey = [0]
                else:
                    hostkey = [0 for ii in xrange(len(shape))]

            if self.fdtd.rank >= i_strt and self.fdtd.rank <= i_stop:
                x_strt_d = x_strt - self.fdtd.x_strt_group[self.fdtd.rank]
                x_stop_d = x_stop - self.fdtd.x_strt_group[self.fdtd.rank]
                x_strt_h = self.fdtd.x_strt_group[self.fdtd.rank] - x_strt
                x_stop_h = self.fdtd.x_stop_group[self.fdtd.rank] - x_strt
                for k in xrange(self.ndim):
                    if k == 0:
                        if   self.fdtd.rank == i_strt and self.fdtd.rank == i_stop:
                            if   isinstance(key[0], int_types): newkey[0] =   int(x_strt_d)
                            elif isinstance(key[0],     slice): newkey[0] = slice(x_strt_d, x_stop_d+1, x_step)
                            hostkey[0] = slice(None,None,None)
                        elif self.fdtd.rank == i_strt:
                            if   isinstance(key[0], int_types): newkey[0] =   int(x_strt_d)
                            elif isinstance(key[0],     slice): newkey[0] = slice(x_strt_d,       None, x_step)
                            hostkey[0] = slice(None,x_stop_h+1,None)
                        elif self.fdtd.rank != i_stop:
                            if   isinstance(key[0], int_types): newkey[0] =   int(       0)
                            elif isinstance(key[0],     slice): newkey[0] = slice(       0,       None, x_step)
                            hostkey[0] = slice(x_strt_h,x_stop_h+1,None)
                        else:
                            if   isinstance(key[0], int_types): newkey[0] =   int(       0)
                            elif isinstance(key[0],     slice): newkey[0] = slice(       0, x_stop_d+1, x_step)
                            hostkey[0] = slice(x_strt_h,None,None)
                    else:
                        if   isinstance(key[k], int_types): newkey[k] =   int(strts[k])
                        elif isinstance(key[k],     slice): newkey[k] = slice(strts[k], stops[k]+1, steps[k])

            if self.fdtd.rank >= i_strt and self.fdtd.rank <= i_stop:
                val  = self.data.__getitem__(newkey)
                if isinstance(val, np.ndarray):
                    if not val.flags['C_CONTIGUOUS']:
                        val = np.ascontiguousarray(val)
                if self.fdtd.master:
                    if shape == []:
                        value = val#*np.ones(1, dtype=type(val))
                    else:
                        value[hostkey] = val
                else:
                    self.fdtd.comm.Send(val, dest=self.fdtd.size, tag=10)

            if self.fdtd.master:
                x_strt_h_group, x_stop_h_group = [np.zeros(self.fdtd.size, dtype=int) for i in xrange(2)]
                for i in xrange(i_strt, i_stop+1):
                    x_strt_h_group[i] = self.fdtd.x_strt_group[i] - x_strt
                    x_stop_h_group[i] = self.fdtd.x_stop_group[i] - x_strt
                if 'ce' in self.name:
                    for i in xrange(i_strt, i_stop+1):
                        if   i == i_strt and i == i_stop:
                            self.fdtd.comm.Recv(value, source=i, tag=10)
                        elif i == i_strt:
                            hostkey[0] = slice(None,x_stop_h_group[i]+1,None)
                            self.fdtd.comm.Recv(value[hostkey], source=i, tag=10)
                        elif i != i_stop:
                            hostkey[0] = slice(x_strt_h_group[i],x_stop_h_group[i]+1,None)
                            self.fdtd.comm.Recv(value[hostkey], source=i, tag=10)
                        else:
                            hostkey[0] = slice(x_strt_h_group[i],None,None)
                            self.fdtd.comm.Recv(value[hostkey], source=i, tag=10)
                else:
                    for i in xrange(i_stop, i_strt-1, -1):
                        if   i == i_strt and i == i_stop:
                            if not isinstance(value, np.ndarray):
                                value = np.zeros(1, dtype=self.dtype)
                                self.fdtd.comm.Recv(value, source=i, tag=10)
                                value = value[0]
                            else:
                                self.fdtd.comm.Recv(value, source=i, tag=10)
                        elif i == i_strt:
                            hostkey[0] = slice(None,x_stop_h_group[i]+1,None)
                            self.fdtd.comm.Recv(value[hostkey], source=i, tag=10)
                        elif i != i_stop:
                            hostkey[0] = slice(x_strt_h_group[i],x_stop_h_group[i]+1,None)
                            self.fdtd.comm.Recv(value[hostkey], source=i, tag=10)
                        else:
                            hostkey[0] = slice(x_strt_h_group[i],None,None)
                            self.fdtd.comm.Recv(value[hostkey], source=i, tag=10)

            '''
            if self.fdtd.master:
                for i in xrange(1, self.fdtd.size):
                    self.fdtd.comm.Send(value, dest=i, tag=10)
            else:
                self.fdtd.comm.Recv(value, source=0, tag=10)
            '''
            #self.fdtd.comm.Bcast(value, root=0)
            return value
        else:
            return self.real.__getitem__(key) + 1.j*self.imag.__getitem__(key)

    def setitem_operation_type(self, key, value, opt):
        self.access = True
        if not self.is_complex:
            strts, stops, steps, n_h = ndarray.to_indices(self, key)
            stops[0] = strts[0] + steps[0]*(n_h[0]-1)
            x_strt = strts[0]
            x_stop = stops[0]
            x_step = steps[0]

            for i in xrange(self.fdtd.size):
                if   x_strt == 0:
                    i_strt = 0
                elif x_strt > self.fdtd.x_strt_group[i] and x_strt <= self.fdtd.x_stop_group[i]:
                    i_strt = i
                if   x_stop >= self.fdtd.x_strt_group[i] and x_stop  <= self.fdtd.x_stop_group[i]:
                    i_stop = i

            shape = []
            for i in xrange(self.ndim):
                if   isinstance(key[i], int_types):
                    pass
                elif isinstance(key[i], slice):
                    shape.append(n_h[i])
            if not isinstance(value, np.ndarray):
                value = np.ones(1, dtype=self.dtype)*value

            hostkey = []
            newkey = [None for ii in xrange(self.ndim)]
            for k in key:
                if   isinstance(k,     slice): hostkey.append(slice(None,None,None))
            if hostkey == []:
                if shape == []:
                    hostkey = [0]
                else:
                    hostkey = [0 for ii in xrange(len(shape))]

            if self.fdtd.rank >= i_strt and self.fdtd.rank <= i_stop:
                x_strt_d = x_strt - self.fdtd.x_strt
                x_stop_d = x_stop - self.fdtd.x_strt
                x_strt_h = self.fdtd.x_strt - x_strt
                x_stop_h = self.fdtd.x_stop - x_strt
                for k in xrange(self.ndim):
                    if k == 0:
                        if   self.fdtd.rank == i_strt and self.fdtd.rank == i_stop:
                            if   isinstance(key[0], int_types): newkey[0] =   int(x_strt_d)
                            elif isinstance(key[0],     slice): newkey[0] = slice(x_strt_d, x_stop_d+1, x_step)
                            hostkey[0] = slice(None,None,None)
                        elif self.fdtd.rank == i_strt:
                            if   isinstance(key[0], int_types): newkey[0] =   int(x_strt_d)
                            elif isinstance(key[0],     slice): newkey[0] = slice(x_strt_d,       None, x_step)
                            hostkey[0] = slice(None,x_stop_h+1,None)
                        elif self.fdtd.rank != i_stop:
                            if   isinstance(key[0], int_types): newkey[0] =   int(       0)
                            elif isinstance(key[0],     slice): newkey[0] = slice(       0,       None, x_step)
                            hostkey[0] = slice(x_strt_h,x_stop_h+1,None)
                        else:
                            if   isinstance(key[0], int_types): newkey[0] =   int(       0)
                            elif isinstance(key[0],     slice): newkey[0] = slice(       0, x_stop_d+1, x_step)
                            hostkey[0] = slice(x_strt_h,None,None)
                    else:
                        if   isinstance(key[k], int_types): newkey[k] = int(strts[k])
                        elif isinstance(key[k],     slice): newkey[k] = slice(strts[k], stops[k]+1, steps[k])
                self.data.setitem_operation_type(newkey, value.__getitem__(hostkey), opt)
        else:
            if isinstance(values, np.ndarray):
                values_real = np.zeros_like(values, dtype=comp_to_real(self.dtype))
                values_imag = np.zeros_like(values, dtype=comp_to_real(self.dtype))
                slices = []
                for i in xrange(self.ndim):
                    slices.append(slice(None,None,None))
                values_real.__setitem__(slices, values.real.__getitem__(slices))
                values_imag.__setitem__(slices, values.imag.__getitem__(slices))
                self.real.setitem_operation_type(key, values_real, operation_type)
                self.imag.setitem_operation_type(key, values_imag, operation_type)
            else:
                self.real.setitem_operation_type(key, values.real, operation_type)
                self.imag.setitem_operation_type(key, values.imag, operation_type)

    def __setitem__(self, key, value):
        self.setitem_operation_type(key, value, 0)

    def set_fields(self, key, value, opt=None):
        opt_to_int = {None:0, '=':0, '+':1, '-':2, '*':3, '/':4}
        opt_int = opt_to_int[opt]
        self.setitem_operation_type(key, value, opt_int)

    def get(self, key):
        if not self.is_complex:
            if self.data.fdtd.extension == 'single':
                self.data.use_buffer_mode = True
            else:
                for data in self.data.datas:
                    data.use_buffer_mode = True
            got = self.__getitem__(key)
            if self.data.fdtd.extension == 'single':
                self.data.use_buffer_mode = self.data.fdtd.ndarray_use_buffer_mode
            else:
                for data in self.data.datas:
                    data.use_buffer_mode = data.fdtd.ndarray_use_buffer_mode
        else:
            got = self.real.get(key) + 1.j*self.imag.get(key)
        return got

    def set(self, key, value):
        if not self.is_complex:
            if self.data.fdtd.extension == 'single':
                self.data.use_buffer_mode = True
            else:
                for data in self.data.datas:
                    data.use_buffer_mode = True
            self.__setitem__(key, value)
            if self.data.fdtd.extension == 'single':
                self.data.use_buffer_mode = self.data.fdtd.ndarray_use_buffer_mode
            else:
                for data in self.data.datas:
                    data.use_buffer_mode = data.fdtd.ndarray_use_buffer_mode
        else:
            value_real = np.ascontiguousarray(np.real(value))
            value_imag = np.ascontiguousarray(np.imag(value))
            self.real.set(key, value_real)
            self.imag.set(key, value_imag)

if __name__ == '__main__':
    import SMS
    import time
    import numpy.linalg as la
    '''
    from opencl import NVIDIA
    str_dtype = 'float'
    activate_float64 = ''
    ctx = cl.Context([NVIDIA.DEVICES[0]])
    template = file(SMS.__file__.rstrip('__init__.pyc') + 'src/get_set_data.c','r').read()
    code  = template_to_code(template, ['LOCAL_SIZE', 'ACTIVATE_FLOAT64', 'FLOAT'], [str(NVIDIA.local_size), activate_float64, str_dtype])
    prg = cl.Program(ctx, code).build()
    queue = cl.CommandQueue(ctx, NVIDIA.DEVICES[0])
    '''
    import SMS.engine as eng
#    nvidia = eng.nvidia_opencl()
#    nvidia = eng.nvidia_cuda()
#    nvidia.test_init()
    intel = eng.intel_cpu()
    intel.test_init()
    a = ndarray((4,5,6), dtype=np.float32)
    b = np.arange(4*5*6, dtype=np.float32).reshape(4,5,6)
    true_a = np.zeros_like(b)
#    a.set_buffer(np.zeros(4*5*6, dtype=np.float32), ctx, queue, prg, NVIDIA.local_size)
#    a.set_buffer(np.zeros(4*5*6, dtype=np.float32), nvidia)
    a.set_buffer(np.zeros(4*5*6, dtype=np.float32), intel)
    print a[:,:,:]
    a[:,:,:] = b[:,:,:]
#    cl.enqueue_read_buffer(a.queue, a.data, true_a).wait()
#    assert la.norm(a[:,:,:]-b[:,:,:]) == 0
#    assert la.norm(true_a-b) == 0
#    print a[0,:,:]
#    print a[:,0,:]
#    print a[:,:,0]
#    print a[:,0,0]
#    print a[0,:,0]
#    print a[0,0,:]
#    print a[0,0,0]
    a[:,:,:] = 0.
    print a[:,:,:].mean()
    a[:,:,:] = 1.
    print a[:,:,:].mean()

    a = ndarray((100,100), dtype=np.float32)
#    a.set_buffer(np.zeros((100,100), dtype=np.float32), ctx, queue, prg, NVIDIA.local_size)
    a.set_buffer(np.zeros((100,100), dtype=np.float32), nvidia)
    a[:,:] = 0.
    print a[:,:].mean()
    a[:,:] = 1.
    print a[:,:].mean()
    '''
    x, y, z = [np.ones(10, dtype=np.float64) for i in xrange(3)]
    fdtd_multidev = SMS.Basic_FDTD('3D', (x,y,z), np.float32, 'multi')
    ex = Fields_multi_devices(fdtd_multidev, (11,11,11), dtype=fdtd_multidev.dtype, name='ex')
    print ex[:,:,:].sum()
    true_aa = np.arange(11*11, dtype=np.float32).reshape(11,11)
    ex[:,:,0] = true_aa
    assert la.norm(ex[:,:,0] - true_aa) == 0
#    for fdtd in fdtd_multidev.fdtd_group:
#        print fdtd.ex[:,:,0]
    ex[-1,-1,0] = 1234
#    print ex[:,:,0]
#    print ex[-1,-1,0]
    ex[:,:,:] = 0.
    print ex[:,:,:].mean()

    x, y, z = [np.ones(10, dtype=np.float64) for i in xrange(3)]
    fdtd_mpi = SMS.Basic_FDTD('3D', (x,y,z), np.float32, 'mpi')
    ex = Fields_MPI(fdtd_mpi, (11,11,11), dtype=fdtd_multidev.dtype, name='ex')
    print ex[:,:,:]
    true_aa = np.arange(11*11, dtype=np.float32).reshape(11,11)
    ex[:,:,0] = true_aa
    assert la.norm(ex[:,:,0] - true_aa) == 0
    for fdtd in fdtd_multidev.fdtd_group:
        print fdtd.ex[:,:,0]
    ex[-1,-1,0] = 1234
    print ex[:,:,0]
    print ex[-1,-1,0]
    '''


