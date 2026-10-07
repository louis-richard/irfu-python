pyrfu.plot
==========

.. module:: pyrfu.plot

.. currentmodule:: pyrfu.plot

Plotting routines for time series, spectrograms, particle distributions and
spacecraft configurations, built on matplotlib.

.. code-block:: python

    import matplotlib.pyplot as plt
    from pyrfu.plot import plot_line, plot_spectr

Time series and spectrograms
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   plot_line
   plot_spectr
   pl_tx
   plot_clines
   span_tint
   zoom
   add_position

Particle distributions
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   plot_projection
   plot_reduced_2d
   plot_ang_ang

Spacecraft and magnetosphere
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mms_pl_config
   plot_magnetosphere

Statistics, contours and surfaces
---------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pl_scatter_matrix
   plot_heatmap
   annotate_heatmap
   plot_contour
   plot_surf
   ion_brazil_plot_thresh

Figure styling
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   use_pyrfu_style
   set_color_cycle
   colorbar
   make_labels
