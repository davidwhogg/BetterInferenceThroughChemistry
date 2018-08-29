#!/usr/bin/env python

from collections import defaultdict
import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# Build the C / Cython code
extensions = []

# Get Gala path, Numpy path
import gala
gala_base_path = os.path.split(gala.__file__)[0]

import numpy
numpy_base_path = os.path.split(numpy.__file__)[0]

cfg = defaultdict(list)
cfg['include_dirs'].append(os.path.join(numpy_base_path, 'core', 'include'))
cfg['include_dirs'].append(os.path.join(gala_base_path, 'potential'))
cfg['extra_compile_args'].append('--std=gnu99')
cfg['sources'].append('chemtrails/potential/cpotential.pyx')
cfg['sources'].append('chemtrails/potential/src/potential.c')
extensions.append(Extension('chemtrails.potential', **cfg))

pkg_data = dict()
pkg_data[""] = ["LICENSE", "AUTHORS"]
pkg_data["chemtrails"] = ["potential/src/*.h", "potential/src/*.c"]

setup(name='chemtrails',
      version='0.1',
      description='TODO',
      install_requires=['numpy', 'astro-gala'],
      author='adrn',
      author_email='adrn@astro.princeton.edu',
      license='MIT',
      url='https://github.com/davidwhogg/ChemicalTangents',
      cmdclass={'build_ext': build_ext},
      packages=["chemtrails", "chemtrails.potential"],
      package_data=pkg_data,
      ext_modules=extensions)
