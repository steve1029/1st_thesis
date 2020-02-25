# -*- coding: utf-8 -*-

try:
    import numpy
except ImportError:
    raise ImportError, 'Python Module \'numpy\' is needed for using FDTD'

import engine
import structures
import materials
from mainfdtd   import *
from incident   import *
from materials  import *
from dispersive import *
from units import to_SI, to_NU, to_epr
