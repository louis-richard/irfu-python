

.. |LicenseMIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
.. _LicenseMIT: https://opensource.org/licenses/MIT

.. |LicenseCC| image:: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
.. _LicenseCC: https://creativecommons.org/licenses/by/4.0/

.. |Maintenance| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
.. _Maintenance: https://github.com/louis-richard/irfu-python/graphs/commit-activity

.. |DocSphinx| image:: https://img.shields.io/static/v1.svg?label=sphinx&message=documentation&color=blue
.. _DocSphinx: https://pyrfu.readthedocs.io

.. |PyPi| image:: https://img.shields.io/badge/install_with-pypi-brightgreen.svg
.. _PyPi: https://test.pypi.org/project/pyrfu/

.. |PyLint| image:: https://img.shields.io/badge/pylint-9.46-brightgreen.svg
.. _PyLint: http://pylint.pycqa.org/en/latest/intro.html


pyRFU
=====

|LicenseMit|_ |LicenseCC|_ |PyPi|_  |DocSphinx|_ |Maintenance|_ |PyLint|_

The Python package ``pyrfu`` is a software based on the IRFU-MATLAB library to work with space data, particularly the
Magnetospheric MultiScale (MMS) mission.

It is distributed under the open-source MIT license.

Full documentation can be found `here <https://pyrfu.readthedocs.io>`_

.. end-marker-intro-do-not-remove


.. start-marker-install-do-not-remove


Instalation
-----------
pyRFU supports Windows, macOS and Linux.

Requirements
************
pyRFU uses packages not included in python3. To get started, install the required packages using :

.. code:: python

    pip install -r requirements.txt

From TestPyPi
*************
pyRFU uses TestPyPI a separate instance of the Python Package index to not affect the real index. To get started, install the pyrfu package using TestPyPI:

.. code:: python

    pip install --index-url https://test.pypi.org/project/ --no-deps pyrfu

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