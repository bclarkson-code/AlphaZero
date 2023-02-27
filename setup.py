from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="AlphaZero",
    version="0.0.1",
    description="A sample implementation of AlphaZero for Draughts (checkers)",
    ext_modules=cythonize("c_draughts.pyx"),
    include_dirs=[numpy.get_include()],
)
