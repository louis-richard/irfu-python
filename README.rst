

.. |LicenseMIT| image:: https://img.shields.io/badge/License-MIT-blue.svg
.. _LicenseMIT: https://opensource.org/licenses/MIT

.. |LicenseCC| image:: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
.. _LicenseCC: https://creativecommons.org/licenses/by/4.0/

.. |Maintenance| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
.. _Maintenance: https://github.com/louis-richard/irfu-python/graphs/commit-activity

.. |Doc| image:: https://readthedocs.org/projects/pyspaceweather/badge/?version=latest
.. _Doc: https://pyrfu.readthedocs.io/en/latest/?badge=latest

.. |DocSphinx| image:: https://img.shields.io/static/v1.svg?label=sphinx&message=documentation&color=blue
.. _DocSphinx: https://pyrfu.readthedocs.io

.. |PyPi| image:: https://img.shields.io/pypi/v/pyrfu.svg?logo=python&logoColor=blue
.. _PyPi: https://pypi.org/project/pyrfu/

.. |VPython| image:: https://img.shields.io/pypi/pyversions/pyrfu.svg?logo=python&logoColor=blue
.. _VPython: https://pypi.org/project/pyrfu/

.. |Lint| image:: https://img.shields.io/badge/pylint-9.55-blue.svg?logo=python&logoColor=blue
.. _Lint: http://pylint.pycqa.org/en/latest/intro.html

.. |Downloads| image:: https://img.shields.io/pypi/dm/pyrfu
.. _Downloads: https://pypi.org/project/pyrfu/


pyRFU
=====

|LicenseMit|_ |LicenseCC|_ |VPython|_ |Lint|_ |PyPi|_ |Downloads|_ |Doc|_

The Python package ``pyrfu`` is a software based on the IRFU-MATLAB library to work with space data, particularly the
Magnetospheric MultiScale (MMS) mission.

It is distributed under the open-source MIT license.

Full documentation can be found `here <https://pyrfu.readthedocs.io>`_

.. end-marker-intro-do-not-remove


Instalation
-----------
.. start-marker-install-do-not-remove

The package `pyrfu` has been tested only for Mac OS. `pyrfu` requires `FFTW3 <http://fftw.org>`_. Install FFTW3
using `homebrew <https://brew.sh>`_ and set temporary environmental variables, such that `pyrfu`
finds fftw:

.. code-block:: bash

    brew install fftw
    export DYLD_LIBRARY_PATH=/usr/local/lib
    export LDFLAGS="-L/usr/local/lib"
    export CFLAGS="-I/usr/local/include"

Using PyPi
**********

``pyrfu`` is available for pip.

.. code-block:: bash

        pip install pyrfu


From sources
************

The sources are located on **GitHub**:

    https://github.com/louis-richard/irfu-python

1) Clone the GitHub repo:

Open a terminal and write the below command to clone in your PC the pyrfu repo:

.. code-block:: bash

    git clone https://github.com/louis-richard/irfu-python.git
    cd pyrfu


2) Create a virtual env

pyrfu needs a significant number of dependencies. The easiest
way to get everything installed is to use a virtual environment.

-  Anaconda

You can create a virtual environment and install all the dependencies using conda_
with the following commands:

.. code-block:: bash

    pip install conda
    conda create -n pyrfu
    source activate pyrfu

.. _conda: http://conda.io/


- Virtual Env

Virtualenv_ can also be used:

.. code-block:: bash

    pip install virtualenv
    virtualenv pyrfu
    source pyrfu/bin/activate

.. _virtualenv: https://virtualenv.pypa.io/en/latest/#


3) Install the package via the commands:

.. code-block:: bash

        python setup.py install


5) (Optional) Generate the docs: install the extra dependencies of doc and run
the `setup.py` file:

.. code-block:: bash

        pip install pyrfu
        python setup.py build_sphinx

Once installed, the doc can be generated with:

.. code-block:: bash

        cd doc
        make html


Dependencies
************

The required dependencies are:

- `Python <https://python.org>`_  >= 3.7
- `python-dateutil <https://dateutil.readthedocs.io/en/stable/>`_ >=2.8.1
- `numpy <https://www.numpy.org>`_ >= 1.18
- `scipy <https://scipy.org>`_ >= 1.4.1
- `matplotlib <https://matplotlib.org>`_ >= 3.2.1
- `pandas <https://pandas.pydata.org/>`_ >= 1.0.3
- `astropy <https://www.astropy.org/>`_ >=4.0.1
- `xarray <https://xarray.pydata.org/en/stable/>`_ >=0.15
- `pyfftw <https://pyfftw.readthedocs.io/en/latest/>`_ >=0.12.0
- `spacepy <https://spacepy.github.io/#>`_ >=0.2.1
- `seaborn <https://seaborn.pydata.org>`_ >=0.10.1
- `sfs <https://sfs-python.readthedocs.io>`_ >=0.5.0
- `tqdm <https://pypi.org/project/tqdm/#documentation>`_ >=4.46.0

Testing dependencies are:

- `pytest <https://docs.pytest.org/en/latest/>`_ >= 2.8

Extra testing dependencies:

- `coverage <https://coverage.readthedocs.io>`_ >= 4.4
- `pylint <https://www.pylint.org>`_ >= 1.6.0

.. end-marker-install-do-not-remove

Usage
-----
To import generic space plasma physics functions

.. code:: python

    from pyrfu import pyrf


To import functions specific to MMS mission

.. code:: python

    from pyrfu import mms


To import plotting functions

.. code:: python

    from pyrfu import plot as pltrf

Configuration
-------------
Configuration settings are set in the CONFIG hash table in the mms_config.py file.

Credits
-------
This software was developped by Louis RICHARD (louisr@irfu.se) based on the IRFU-MATLAB library.

Acknowledgement
---------------
Please use the following to acknowledge use of pyrfu in your publications:
Data analysis was performed using the pyrfu analysis package available at https://github.com/louis-richard/irfu-python

Additional Information
----------------------
MMS Science Data Center: https://lasp.colorado.edu/mms/sdc/public/

MMS Datasets: https://lasp.colorado.edu/mms/sdc/public/datasets/

MMS - Goddard Space Flight Center: http://mms.gsfc.nasa.gov/
