.. -*- mode: rst -*-

pyRFU
=====

.. start-marker-intro-do-not-remove

.. |License| image:: https://img.shields.io/pypi/l/pyrfu
.. _License: https://opensource.org/licenses/MIT

.. |Python| image:: https://img.shields.io/pypi/pyversions/pyrfu.svg?logo=python
.. _Python: https://pypi.org/project/pyrfu/

.. |PyPi| image:: https://img.shields.io/pypi/v/pyrfu.svg?logo=pypi
.. _PyPi: https://pypi.org/project/pyrfu/

.. |Format| image:: https://img.shields.io/pypi/format/pyrfu?color=blue&logo=pypi
.. _Format: https://pypi.org/project/pyrfu/

.. |Wheel| image:: https://img.shields.io/pypi/wheel/pyrfu?logo=pypi&color=blue
.. _Wheel: https://pypi.org/project/pyrfu/

.. |Status| image:: https://img.shields.io/pypi/status/pyrfu?logo=pypi&color=blue
.. _Status: https://pypi.org/project/pyrfu/

.. |Downloads| image:: https://img.shields.io/pypi/dm/pyrfu?logo=pypi&color=blue
.. _Downloads: https://pypi.org/project/pyrfu/

.. |CI| image:: https://github.com/louis-richard/irfu-python/actions/workflows/tests.yml/badge.svg
.. _CI: https://github.com/louis-richard/irfu-python/actions/workflows/tests.yml

.. |PyLintB| image:: https://github.com/louis-richard/irfu-python/actions/workflows/pylint.yml/badge.svg
.. _PyLintB: https://github.com/louis-richard/irfu-python/actions/workflows/pylint.yml

.. |Issues| image:: https://img.shields.io/github/issues/louis-richard/irfu-python?logo=github&color=9cf
.. _Issues: https://github.com/louis-richard/irfu-python/issues

.. |Commits| image:: https://img.shields.io/github/last-commit/louis-richard/irfu-python?logo=github&color=9cf
.. _Commits: https://github.com/louis-richard/irfu-python/commits/master

.. |Readthedocs| image:: https://img.shields.io/readthedocs/pyrfu?logo=read-the-docs&color=blueviolet
.. _Readthedocs: https://pyrfu.readthedocs.io/en/latest/

.. |Gitter| image:: https://img.shields.io/gitter/room/louis-richard/pyrfu?logo=gitter&color=orange
.. _Gitter: https://gitter.im/pyrfu

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

|License|_ |Python|_ |PyPi|_ |Format|_ |Wheel|_ |Status|_ |Downloads|_ |CI|_
|PyLintB|_ |Issues|_ |Commits|_ |Readthedocs|_ |Gitter|_ |Black|_

The Python package ``pyrfu`` is a software based on the IRFU-MATLAB library to work with space data, particularly the
Magnetospheric MultiScale (MMS) mission.

It is distributed under the open-source MIT license.

.. end-marker-intro-do-not-remove

Full documentation can be found `here <https://pyrfu.readthedocs.io>`_


Quickstart
==========

Installing pyrfu with pip (`more details here`_):

.. _more details here: https://pyrfu.readthedocs.io/en/latest/installation.html

.. code-block:: console

    $ python -m pip install pyrfu
    # or
    $ python -m pip install --user pyrfu

Import `pyrfu.mms`_ package with routines specific to work with the
Magnetospheric Multiscale mission (MMS)

.. _pyrfu.mms: https://pyrfu.readthedocs.io/en/latest/dev/pyrfu.mms.html

.. code:: python

    from pyrfu import mms

Setup path to MMS data

.. code:: python

    mms.db_init("/Volumes/mms")

Load magnetic field and ion bulk velocity data

.. code:: python

    tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
    b_gsm = mms.get_data("b_gsm_fgm_srvy_l2", tint, 1)
    v_gse_i = mms.get_data("vi_gse_fpi_fast_l2", tint, 1)

Import `pyrfu.pyrf`_ package with generic routines

.. _pyrfu.pyrf: https://pyrfu.readthedocs.io/en/latest/dev/pyrfu.pyrf.html

.. code:: python

    from pyrfu import pyrf

Transform ion bulk velocity to geocentric solar magnetospheric (GSM) coordinates

.. code:: python

    v_gsm_i = pyrf.cotrans(v_gse_i, "gse>gsm")

Import `pyrfu.plot`_ package with plotting routines

.. _pyrfu.plot: https://pyrfu.readthedocs.io/en/latest/dev/pyrfu.plot.html

.. code:: python

    from pyrfu import plot

Plot time series of magnetic field and ion bulk velocity

.. code:: python

    import matplotlib.pyplot as plt

    f, axs = plt.subplots(2, sharex="all")
    plot.plot_line(axs[0], b_gsm)
    axs[0].set_ylabel("$B~[\\mathrm{nT}]$")
    axs[0].legend(["$B_{x}$", "$B_{y}$", "$B_{z}$"], ncol=4)

    plot.plot_line(axs[1], v_gsm_i)
    axs[1].set_ylabel("$V_i~[\\mathrm{km}~\\mathrm{s}^{-1}]$")
    axs[1].legend(["$V_{ix}$", "$V_{iy}$", "$V_{iz}$"], ncol=4)


Credits
-------
This software was developed by Louis RICHARD (louisr@irfu.se) based on the IRFU-MATLAB library.

Acknowledgement
---------------
Please use the following to acknowledge use of pyrfu in your publications:
Data analysis was performed using the pyrfu analysis package available at https://github.com/louis-richard/irfu-python

Additional Information
----------------------
MMS Science Data Center: https://lasp.colorado.edu/mms/sdc/public/

MMS Datasets: https://lasp.colorado.edu/mms/sdc/public/datasets/

MMS - Goddard Space Flight Center: http://mms.gsfc.nasa.gov/
