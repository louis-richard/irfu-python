:html_theme.sidebar_secondary.remove:

.. rst-class:: pyrfu-landing

pyrfu
=====

.. grid:: 1 1 2 2
   :gutter: 4
   :class-container: pyrfu-hero

   .. grid-item::
      :child-align: center

      .. rst-class:: pyrfu-tagline

      Space plasma physics in Python

      .. rst-class:: pyrfu-lead

      ``pyrfu`` is an open-source package for in-situ space plasma data
      analysis, based on the IRFU-MATLAB library. It covers the
      Magnetospheric Multiscale (MMS) mission from data access to particle
      distributions and wave analysis, and general plasma physics tools.

      .. code-block:: console

         $ python -m pip install pyrfu

      .. container:: pyrfu-buttons

         .. button-ref:: examples/00_overview/quick-overview
            :ref-type: doc
            :color: primary

            Get started

         .. button-ref:: examples/index
            :ref-type: doc
            :color: primary
            :outline:

            Examples

         .. button-ref:: api/index
            :ref-type: doc
            :color: primary
            :outline:

            API reference

   .. grid-item::
      :child-align: center

      .. image:: _static/landing-waves.png
         :alt: Wavelet spectra of the magnetic and electric fields and wave
               ellipticity computed with pyrfu from MMS burst data.
         :target: examples/01_mms/example_mms_polarizationanalysis.html

|PyPI| |Python| |CI| |DOI| |License|

.. |PyPI| image:: https://img.shields.io/pypi/v/pyrfu.svg?logo=pypi
   :target: https://pypi.org/project/pyrfu/
   :alt: PyPI version

.. |Python| image:: https://img.shields.io/pypi/pyversions/pyrfu.svg?logo=python
   :target: https://pypi.org/project/pyrfu/
   :alt: Supported Python versions

.. |CI| image:: https://github.com/louis-richard/irfu-python/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/louis-richard/irfu-python/actions/workflows/tests.yml
   :alt: Continuous integration

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10678695.svg
   :target: https://doi.org/10.5281/zenodo.10678695
   :alt: DOI

.. |License| image:: https://img.shields.io/pypi/l/pyrfu
   :target: https://opensource.org/licenses/MIT
   :alt: MIT license

What you can do with pyrfu
--------------------------

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: pyrfu-cards

   .. grid-item-card:: :octicon:`database` MMS data access
      :link: api-mms-loading
      :link-type: ref

      Find, download and load MMS data from a local archive, the SDC or AWS
      in one call, as ``xarray`` time series.

   .. grid-item-card:: :octicon:`graph` Particle distributions
      :link: api-mms-fpi
      :link-type: ref

      FPI skymaps: omni-directional and pitch-angle distributions,
      projections, reduced 1D and 2D distributions and moments.

   .. grid-item-card:: :octicon:`pulse` Waves and polarization
      :link: api-pyrf-waves
      :link-type: ref

      Wavelet and Fourier spectra, polarization analysis, Poynting flux and
      four-spacecraft dispersion relations.

   .. grid-item-card:: :octicon:`git-merge` Multi-spacecraft methods
      :link: api-pyrf-multi-sc
      :link-type: ref

      Curlometer current density, gradients, timing velocities and
      four-spacecraft averages.

   .. grid-item-card:: :octicon:`globe` Coordinates and time series
      :link: api-pyrf-coordinates
      :link-type: ref

      Geophysical coordinate transformations, field-aligned and minimum
      variance frames, filtering and resampling.

   .. grid-item-card:: :octicon:`paintbrush` Plotting
      :link: api/plot
      :link-type: doc

      Publication-ready time series, spectrograms and particle distribution
      plots with matplotlib.

Quickstart
----------

.. code-block:: python

    import matplotlib.pyplot as plt
    from pyrfu import mms, pyrf
    from pyrfu.plot import plot_line

    # Path to the MMS data
    mms.db_init(default="local", local="/Volumes/mms")

    # Load the magnetic field and the ion bulk velocity
    tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
    b_gsm = mms.get_data("b_gsm_fgm_srvy_l2", tint, 1)
    v_gse_i = mms.get_data("vi_gse_fpi_fast_l2", tint, 1)

    # Transform the ion bulk velocity to GSM coordinates
    v_gsm_i = pyrf.cotrans(v_gse_i, "gse>gsm")

    f, axs = plt.subplots(2, sharex="all")
    plot_line(axs[0], b_gsm)
    plot_line(axs[1], v_gsm_i)

More in the :doc:`examples gallery <examples/index>`.

Documentation
-------------

.. grid:: 1 2 4 4
   :gutter: 3
   :class-container: pyrfu-cards

   .. grid-item-card:: :octicon:`download` Installation
      :link: installation
      :link-type: doc

      Install from PyPI or from the sources.

   .. grid-item-card:: :octicon:`book` Examples
      :link: examples/index
      :link-type: doc

      Notebooks for MMS, dispersion relations and more.

   .. grid-item-card:: :octicon:`code` API reference
      :link: api/index
      :link-type: doc

      Every function, grouped by topic.

   .. grid-item-card:: :octicon:`people` Contributing
      :link: contributing
      :link-type: doc

      Code style, tests and how to contribute.

Citing pyrfu
------------

If you use ``pyrfu`` in your research, please cite it using its Zenodo DOI
`10.5281/zenodo.10678695 <https://doi.org/10.5281/zenodo.10678695>`_ (the
Zenodo page also lists the DOI of each release):

.. code-block:: bibtex

    @software{richard2024pyrfu,
      author    = {Richard, Louis and Khotyaintsev, Yuri V. and Vaivads, Andris and
                   Graham, Daniel B. and Norgren, Cecilia and Johlander, Andreas},
      title     = {Python RymdFysik Utilities (PyRFU): An Open-Source Python Package
                   for Advanced In-Situ Space Plasma Analysis},
      year      = {2024},
      publisher = {Zenodo},
      doi       = {10.5281/zenodo.10678695},
      url       = {https://doi.org/10.5281/zenodo.10678695}
    }

``pyrfu`` was developed at the Swedish Institute of Space Physics (IRF) in
Uppsala, building on the IRFU-MATLAB library, and is now maintained at the
Rudolf Peierls Centre for Theoretical Physics, University of Oxford. It is
distributed under the MIT license.

.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   examples/index
   api/index
   contributing
