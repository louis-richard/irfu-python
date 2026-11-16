.. highlight:: shell

============
Installation
============


Stable release
--------------

To install Pyrfu, run this command in your terminal:

.. code-block:: console

    $ python -m pip install pyrfu
    # or
    $ python -m pip install --user pyrfu

This is the preferred method to install Pyrfu, as it will always install the most
recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for Pyrfu can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/louis-richard/irfu-python

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python -m pip install .


.. _Github repo: https://github.com/louis-richard/irfu-python


Known issues
------------

With numpy 2.0 to 2.2 on macOS (Apple Accelerate), matrix products of finite
arrays can give false warnings, e.g. ``RuntimeWarning: divide by zero
encountered in matmul``, in routines such as ``pyrf.mva``, ``pyrf.new_xyz`` or
``mms.vdf_shift``. The results are correct; numpy 2.3 and later no longer give
them:

.. code-block:: console

    $ python -m pip install --upgrade "numpy>=2.3"
