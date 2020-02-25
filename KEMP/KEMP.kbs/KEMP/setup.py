try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name="KMES", \
      version="0.1", \
      py_modules=["SMS_2013"], \
      scripts=[ "__init__.py", \
                "mainfdtd.py", \
                "ndarray.py", \
                "engine.py", \
                "util.py", \
                "units.py", \
                "exception.py", \
                "pml.py", \
                "pbc.py", \
                "incident.py", \
                "rft.py", \
                "dispersive.py", \
                "materials.py", \
                "structures.py" \
               ], \
      data_files=([("src_cpu", ["src/cpu/fdtd.c", \
                                "src/cpu/cpml.c", \
                                "src/cpu/pbc.c", \
                                "src/cpu/bbc.c", \
                                "src/cpu/tfsf.c", \
                                "src/cpu/dispersive.c", \
                                "src/cpu/running_fourier_transform.c", \
                                "src/cpu/structures.c" \
                               ],
                   ), \
                   ("src_nvd", ["src/nvidia/fdtd", \
                                "src/nvidia/get_set_data", \
                                "src/nvidia/incident_direct", \
                                "src/nvidia/cpml", \
                                "src/nvidia/pbc", \
                                "src/nvidia/bbc", \
                                "src/nvidia/tfsf", \
                                "src/nvidia/dispersive", \
                                "src/nvidia/running_fourier_transform", \
                                "src/nvidia/structures", \
                               ], \
                   ), \
                  ] \
                 ) \
     )

