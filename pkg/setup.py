#!/usr/bin/env python

from distutils.core import setup

pkg_data = {'chemtrails': 'tests/galah-small.fits'}

setup(name='chemtrails',
      version='0.1',
      description='TODO',
      install_requires=['numpy', 'astro-gala'],
      author='adrn',
      author_email='adrn@astro.princeton.edu',
      license='MIT',
      url='https://github.com/davidwhogg/ChemicalTangents',
      package_data=pkg_data,
      packages=["chemtrails", "chemtrails.potential"])
