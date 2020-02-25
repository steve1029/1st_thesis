# Author  : Ki-Hwan Kim
# Purpose : Common utilities
# Target  : CPU using C
# Created : 2012-02-09
# Modified: 

import imp
import numpy as np
import os
import platform
import subprocess as sp
import tempfile
import unittest

import common


sep = os.path.sep
src_path = sep.join(common.__file__.split(sep)[:-2] + ['cpu', 'src', ''])

system = platform.system()
pyver = platform.python_version_tuple()[:2]
sopath = lambda cpath: {'Linux': cpath.replace('.c', '.so'), \
                        'Windows': cpath.replace('.c', '.pyd') }[system]


def print_cpu_info():
    if system == 'Linux':
        for line in open('/proc/cpuinfo'):
            if 'model name' in line:
                cpu_name0 = line[line.find(':')+1:-1]
                break
        cpu_name = ' '.join(cpu_name0.split())
    
        for line in open('/proc/meminfo'):
            if 'MemTotal' in line:
                mem_nbytes = int(line[line.find(':')+1:line.rfind('kB')]) * 1024
                break
        print('Host Device :')
        print('  name: %s' % cpu_name)
        print('  mem size: %1.2f %s' % common.binary_prefix_nbytes(mem_nbytes))
        print('')

        

class CompileError(Exception):
    """
    User-defined exception
    Return the compile error message
    """

    def __init__(self, stderr):
        Exception.__init__(self)
        self.stderr = stderr


    def __str__(self):
        return '\n' + self.stderr



def build_clib(src, libname):
    """
    Build the C code and Return the python module
    """

    path = tempfile.gettempdir() + '/kemp'
    if not os.path.exists(path): 
        os.mkdir(path)
    cfile = tempfile.NamedTemporaryFile(suffix='.c', dir=path, delete=False)
    cfile.write(src)
    cfile.close()
    
    cpath = cfile.name
    libpath = sopath(cpath)
    ILpath = {'Linux': '-fpic -I/usr/include/pythonVER'.replace('VER','.'.join(pyver)), \
              'Windows': '-IC:\PythonVER/include -IC:\PythonVER/Lib/site-packages/numpy/core/include -LC:\PythonVER/libs'.replace('VER',''.join(pyver))}[system]
    pylib = {'Linux': '', \
             'Windows': '-lpythonVER'.replace('VER',''.join(pyver))}[system]
    cmd = 'gcc -O3 -std=c99 -Wall -fopenmp -msse %s %s %s -shared -o %s' % (ILpath, cpath, pylib, libpath)
    proc = sp.Popen(cmd.split(), stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    if stderr != '':
        print src
        raise CompileError(stderr)

    try:
        program = imp.load_dynamic(libname, libpath)
    except:
        print('import error')

    return program



class TestFunctions(unittest.TestCase):
    def test_print_cpu_info(self):
        print_cpu_info()


    def test_build_clib(self):
        src = """
        #include <Python.h>
        #include <numpy/arrayobject.h>
        #include <omp.h>
        #include <xmmintrin.h>
        #define LOAD _mm_load_ps
        #define STORE _mm_store_ps
        #define ADD _mm_add_ps
        
        static PyObject *vecadd(PyObject *self, PyObject *args) {
            PyArrayObject *A, *B, *C;
            if(!PyArg_ParseTuple(args, "OOO", &A, &B, &C)) return NULL;
            
            int nx, idx;
            float *a, *b, *c;
            nx = (int)(A->dimensions)[0];
            a = (float*)(A->data);
            b = (float*)(B->data);
            c = (float*)(C->data);
            
            __m128 xa, xb;
            
            Py_BEGIN_ALLOW_THREADS
            #pragma omp parallel for private(idx, xa, xb)
            for(idx=0; idx<nx; idx+=4) {
                xa = LOAD(a+idx);
                xb = LOAD(b+idx);
                STORE(c+idx, ADD(xa, xb));
            }
            Py_END_ALLOW_THREADS
            Py_INCREF(Py_None);
            return Py_None;
        }
        
        static PyMethodDef ufunc_methods[] = {
            {"vecadd", vecadd, METH_VARARGS, ""}, 
            {NULL, NULL, 0, NULL}
        };
        
        PyMODINIT_FUNC initvecop() {
            Py_InitModule("vecop", ufunc_methods);
            import_array();
        }
        """
        program = build_clib(src, 'vecop')

        nx = 4**10
        a = np.random.rand(nx).astype(np.float32)
        b = np.random.rand(nx).astype(np.float32)
        c = np.zeros_like(a)
        program.vecadd(a, b, c)
        self.assertEqual(np.linalg.norm(c - (a+b)), 0)



if __name__ == '__main__':
    unittest.main()
