import os
import sys

class EXITsignal(Exception):
    pass

class FDTDError(Exception):
    pass

class ComputingEngineError(FDTDError):
    pass

class DeviceMemoryError(FDTDError):
    pass

class BoundaryConditionError(FDTDError):
    pass

class SourceConditionError(FDTDError):
    pass

class NonUniformGridError(FDTDError):
    pass
