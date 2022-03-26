
.. |Logo| image:: docs/source/_static/logo-pyrfu.png
    :target: https://pypi.org/project/pyrfu/

.. |License| image:: https://img.shields.io/pypi/l/pyrfu
    :target: https://opensource.org/licenses/MIT

.. |Python| image:: https://img.shields.io/pypi/pyversions/pyrfu.svg?logo=python
    :target: https://pypi.org/project/pyrfu/

.. |PyPi| image:: https://img.shields.io/pypi/v/pyrfu.svg?logo=pypi
    :target: https://pypi.org/project/pyrfu/

.. |Format| image:: https://img.shields.io/pypi/format/pyrfu?color=blue&logo=pypi
    :target: https://pypi.org/project/pyrfu/

.. |Wheel| image:: https://img.shields.io/pypi/wheel/pyrfu?logo=pypi&color=blue
    :target: https://pypi.org/project/pyrfu/

.. |Status| image:: https://img.shields.io/pypi/status/pyrfu?logo=pypi&color=blue
    :target: https://pypi.org/project/pyrfu/

.. |Downloads| image:: https://img.shields.io/pypi/dm/pyrfu?logo=pypi&color=blue
    :target: https://pypi.org/project/pyrfu/

.. |ScrutinizerBuild| image:: https://img.shields.io/scrutinizer/build/g/louis-richard/irfu-python?logo=scrutinizer-ci
    :target: https://scrutinizer-ci.com/g/louis-richard/irfu-python/

.. |ScrutinizerQuality| image:: https://img.shields.io/scrutinizer/quality/g/louis-richard/irfu-python?logo=scrutinizer-ci
    :target: https://scrutinizer-ci.com/g/louis-richard/irfu-python/

.. |Issues| image:: https://img.shields.io/github/issues/louis-richard/irfu-python?logo=github&color=9cf
    :target: https://github.com/louis-richard/irfu-python/issues

.. |Commits| image:: https://img.shields.io/github/last-commit/louis-richard/irfu-python?logo=github&color=9cf
    :target: https://github.com/louis-richard/irfu-python/commits/master

.. |Readthedocs| image:: https://img.shields.io/readthedocs/pyrfu?logo=read-the-docs&color=blueviolet
    :target: https://pyrfu.readthedocs.io/en/latest/

.. |Gitter| image:: https://img.shields.io/gitter/room/louis-richard/pyrfu?logo=gitter&color=orange
    :target: https://gitter.im/pyrfu


|Logo|

pyRFU
=====
.. start-marker-intro-do-not-remove

|License| |Python| |PyPi| |Format| |Wheel| |Status| |Downloads| |ScrutinizerBuild|
|ScrutinizerQuality| |Commits| |Issues| |Readthedocs| |Gitter|

The Python package ``pyrfu`` is a software based on the IRFU-MATLAB library to work with space data, particularly the
Magnetospheric MultiScale (MMS) mission.

It is distributed under the open-source MIT license.

.. end-marker-intro-do-not-remove

Full documentation can be found `here <https://pyrfu.readthedocs.io>`_


Instalation
-----------
.. start-marker-install-do-not-remove

The package ``pyrfu`` has been tested only for Mac OS.

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

Open a terminal and write the below command to clone in your PC the ``pyrfu`` repo:

.. code-block:: bash

    git clone https://github.com/louis-richard/irfu-python.git
    cd irfu-python


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

- `cdflib <https://cdflib.readthedocs.io/en/latest/?badge=latest>`_ >=0.3.20
- `matplotlib <https://matplotlib.org>`_ >=3.2.1
- `numba <http://numba.pydata.org>`_ >=0.52.0
- `numpy <https://www.numpy.org>`_ >=1.18.1
- `pandas <https://pandas.pydata.org/>`_ >=1.0.3
- `Sphinx <https://www.sphinx-doc.org/en/master/>`_ >=3.0.3
- `scipy <https://scipy.org>`_ >=1.4.1
- `setuptools <https://packaging.python.org/tutorials/installing-packages/>`_ >=47.3.1
- `xarray <https://xarray.pydata.org/en/stable/>`_ >=0.15.0


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
Default configuration settings for MMS data (i.e data path) are stored in pyrfu/mms/config.json and can be changed at anytime using mms.db_init(local_path_dir).

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
