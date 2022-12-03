#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import platform
import numpy

# dependencies
install_reqs = [
    'cmake>=3.13',
    'ipython',
    'keras==2.8.0',
    'numpy>=1.17.0',
    'pandas==1.4.2',
    'pathos==0.2.9',
    'pytest==7.1.2',
    'snakemake==5.31.1',
    'tables',
    'pysam==0.19.1',
    'protobuf==3.20',
    'tensorflow==2.8.1',
    'tensorboard<2.9.0',
    'scikit-learn==1.1.1',
    'toolz==0.11.2'
]

compile_args = ['-std=c++11']
link_args = ['-std=c++11']
if platform.system() != 'Darwin':
    compile_args.append('-fopenmp')
    link_args.append('-fopenmp')

sourcefiles = ['resmico/reader.pyx', 'resmico/contig_reader.cpp']

with open("README.md", 'r') as f:
    long_description = f.read()

extensions = [Extension('resmico.reader',
                        sourcefiles,
                        language="c++",
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        libraries=['z'],
                        )]


class get_numpy_include(object):
    """
    Returns Numpy's include path with lazy import.
    """

    def __str__(self):
        import numpy
        return numpy.get_include()


## install main application
desc = 'Increasing the quality of metagenome-assembled genomes with deep learning'
setup(
    name='resmico',
    version='1.2.1',
    description=desc,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Nick Youngblut, Olga Mineeva, Daniel Danciu',
    author_email='nyoungb2@gmail.com',
    entry_points={
        'console_scripts': [
            'resmico = resmico.__main__:main'
        ]
    },
    ext_modules=cythonize(extensions, include_path=['resmico/', numpy.get_include()], annotate=True,
                          compiler_directives={'language_level': "3"}),
    install_requires=install_reqs,
    include_dirs=[get_numpy_include()],
    license="MIT license",
    packages=find_packages(),
    package_data={'resmico': ['reader.pyx', 'contig_reader.hpp', 'contig_reader.cpp',
                              'bam2feat', 'model/resmico.h5', 'model/stats_cov.json']},
    package_dir={'resmico': 'resmico'},
    python_requires='>=3.8',
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'cython',
        'numpy',
    ],
    url='https://github.com/leylabmpi/resmico'
)
