# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
import platform

compile_args = ['-std=c++11']
link_args = ['-std=c++11']
if platform.system() != 'Darwin':
    compile_args.append('-fopenmp')
    link_args.append('-fopenmp')

sourcefiles = ['reader.pyx', 'contig_reader.cpp']

extensions = [Extension('reader',
                        sourcefiles,
                        language="c++",
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        libraries=['z'],
                        )]

setup(
    ext_modules=cythonize(extensions,
                          include_path=['.', numpy.get_include()],
                          annotate=True,
                          compiler_directives={'language_level': "3"}
                          ),
    include_dirs=['.', numpy.get_include()],
)
