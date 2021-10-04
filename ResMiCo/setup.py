# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

sourcefiles = ['Reader.pyx', 'contig_reader.cpp']

extensions = [Extension('Reader',
                        sourcefiles,
                        language="c++",
                        extra_compile_args=['-std=c++11', '-fopenmp'],
                        extra_link_args=['-std=c++11', '-fopenmp'],
                        libraries= ['z'],
                        )]

setup(
    ext_modules=cythonize(extensions, include_path=['.', numpy.get_include()], annotate=True),
    include_dirs=['.', numpy.get_include()],
)
