import pathlib
from setuptools import setup, find_packages
from sphinx.setup_command import BuildDoc

HERE = pathlib.Path(__file__).parent

VERSION = "1.8.5"
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
      "python-dateutil",
      "matplotlib",
      "numpy",
      "sphinx>=1.4",
      "pandas",
      "pvlib",
      "pyfftw",
      "scipy",
      "seaborn",
      "sfs",
      "tqdm",
      "xarray",
      "sphinx_rtd_theme", 
      "ipython", 
      "ipykernel", 
      "nbsphinx",
      "spacepy",
      "sphinxcontrib-apidoc"]

PYTHON_REQUIRES = '>=3.7'


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
