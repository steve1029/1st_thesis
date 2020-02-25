# Author  : Myung-Su Seok
# Purpose : Python wrappers of the active materials of FDTD
# Target  : CPU, GPU

# Python modules
import numpy as np
from scipy.constants import c as c0, hbar

from ndarray import Fields
from mainfdtd import Basic_FDTD
from units import to_SI, to_NU
from util  import *

# CLASSES for Active Materials
class Active_2d:
    def __init__(self, fdtd):
        self.fdtd = fdtd
        self.fdtd.cores['active'] = self
        self.fdtd.cores['pole_active'] = self
        self.fdtd.engine.updates['pole_active'] = self.update_pole
        self.fdtd.engine.updates['active'] = self.update_e
        self.materials = []

class Active_3d:
    def __init__(self, fdtd):
        self.fdtd = fdtd
        self.fdtd.cores['active'] = self
        self.fdtd.cores['pole_active'] = self
        self.fdtd.engine.updates['pole_active'] = self.update_pole
        self.fdtd.engine.updates['active'] = self.update_e
        self.materials = []

class Active_4_levels:
    def __init__(self, taus, wfreqs, gammas, sigmas):
        self.tau_10, self.tau_21, self.tau_32 = taus
        self.wfreq_a, self.wfreq_b = wfreqs
        self.gamma_a, self.gamma_b = gammas
        self.sigma_a, self.sigma_b = sigmas

class Active_2_levels:
    def __init__(self):
        pass

class Active_material:
    def __init__(self):
        pass
