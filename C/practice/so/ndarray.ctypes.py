import ctypes as cty
import numpy as np

a = np.arange(5)

lib = ctypes.cdll.LoadLibrary("libnumcty.so")


