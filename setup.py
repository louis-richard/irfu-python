#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2020 Louis Richard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

import pathlib
from setuptools import setup, find_packages
from sphinx.setup_command import BuildDoc

HERE = pathlib.Path(__file__).parent

VERSION = "2.3.6"
PACKAGE_NAME = "pyrfu"
AUTHOR = "Louis RICHARD"
AUTHOR_EMAIL = "louir@irfu.se"
URL = "https://github.com/louis-richard/irfu-python"

LICENSE = "MIT License"
DESCRIPTION = "Python Space Physics Environment Data Analysis"

with open("README.rst", "r") as fh:
    LONG_DESCRIPTION = fh.read()


INSTALL_REQUIRES = [
      "astropy",
      "cycler",
      "cdflib",
      "cython",
      "ipykernel",
      "ipython",
      "matplotlib",
      "nbsphinx",
      "numpy",
      "numba",
      "pandas",
      "pvlib",
      "pyfftw",
      "python-dateutil",
      "scipy",
      "seaborn",
      "sfs",
      "sphinx>=1.4",
      "sphinxcontrib-apidoc",
      "pydata-sphinx-theme",
      "tqdm",
      "xarray"]

PYTHON_REQUIRES = '>=3.7, <=3.9'


cmdclass = {'build_sphinx': BuildDoc}

setup(
      name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      python_requires=PYTHON_REQUIRES,
      packages=find_packages(),
      include_package_data=True,
      cmdclass=cmdclass,
      # these are optional and override conf.py settings
      command_options={
        'build_sphinx': {
            'project': ('setup.py', PACKAGE_NAME),
            'version': ('setup.py', VERSION),
            'release': ('setup.py', VERSION),
            'source_dir': ('setup.py', 'docs')}},
      )
