import h5py as _h5
import numpy as _np

class export_to_h5:
    def __init__(self, filename):
        self._f = _h5.File(filename, 'w')
