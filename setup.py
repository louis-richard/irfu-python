#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import pathlib

from setuptools import setup, find_packages
from sphinx.setup_command import BuildDoc

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2022"
__license__ = "MIT"
__version__ = "2.3.16"
__status__ = "Prototype"

HERE = pathlib.Path(__file__).parent

VERSION = __version__
PACKAGE_NAME = "pyrfu"
AUTHOR = "Louis RICHARD"
AUTHOR_EMAIL = "louir@irfu.se"
URL = "https://github.com/louis-richard/irfu-python"

LICENSE = "MIT License"
DESCRIPTION = "Python Space Physics (RymdFysik) Utilities"

with open("README.rst", "r") as fh:
    LONG_DESCRIPTION = fh.read()


INSTALL_REQUIRES = ["cdflib", "matplotlib", "numpy", "numba", "pandas",
                    "sphinx", "scipy", "tqdm", "xarray", "geopack"]

PYTHON_REQUIRES = ">=3.7"


cmdclass = {"build_sphinx": BuildDoc}

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
        "build_sphinx": {
            "project": ("setup.py", PACKAGE_NAME),
            "version": ("setup.py", VERSION),
            "release": ("setup.py", VERSION),
            "source_dir": ("setup.py", "docs")}},
      )
