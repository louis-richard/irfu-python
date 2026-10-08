pyrfu.pyrf
==========

.. module:: pyrfu.pyrf

.. currentmodule:: pyrfu.pyrf

Generic routines to build, transform and analyse space plasma time series:
time series construction, time conversions, coordinate systems, signal
processing, wave analysis, multi-spacecraft methods and plasma parameters.

.. code-block:: python

    from pyrfu import pyrf

Time series construction
------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ts_scalar
   ts_vec_xyz
   ts_tensor_xyz
   ts_spectr
   ts_skymap
   ts_time
   ts_append
   dist_append

Time series utilities
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   time_clip
   resample
   t_eval
   start
   end
   extend_tint
   calc_dt
   calc_fs
   find_closest
   remove_repeated_points
   date_str

Time format conversions
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cdfepoch2datetime64
   datetime2iso8601
   datetime642iso8601
   datetime642ttns
   datetime642unix
   iso86012datetime
   iso86012datetime64
   iso86012timevec
   iso86012unix
   timevec2iso8601
   ttns2datetime64
   unix2datetime64

Vector and tensor operations
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dot
   cross
   norm
   normalize
   trace
   dec_par_perp
   cart2sph
   cart2sph_ts
   sph2cart
   solid_angle

.. _api-pyrf-coordinates:

Coordinate systems
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cotrans
   gse2gsm
   convert_fac
   new_xyz
   mva
   mva_gui
   mean
   eb_nrf
   l_shell

Filtering and signal processing
-------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   filt
   lowpass
   medfilt
   movmean
   mean_field
   ts_convolve
   integrate
   gradient
   sliding_derivative
   autocorr
   corr_deriv

.. _api-pyrf-waves:

Spectral and wave analysis
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   psd
   wave_fft
   wavelet
   compress_cwt
   ebsp
   wavepolarize_means
   poynting_flux
   match_phibe_dir
   match_phibe_v

Turbulence and intermittency
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   increments
   struct_func
   pvi

.. _api-pyrf-multi-sc:

Multi-spacecraft methods
------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   avg_4sc
   nanavg_4sc
   c_4_k
   c_4_grad
   c_4_j
   c_4_v
   st_diff
   pid_4sc
   pvi_4sc

Plasma parameters
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   plasma_calc
   iplasma_calc
   plasma_beta
   dynamic_press
   e_vxb
   edb
   vht
   estimate

Pressure tensor, anisotropy and agyrotropy
------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pres_anis
   anisotropy_thresholds
   calc_ag
   calc_agyro
   calc_dng
   calc_sqrtq

Velocity distribution functions
-------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   average_vdf
   int_sph_dist

Shocks and boundaries
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   shock_normal
   shock_parameters
   magnetosphere

Statistics and histograms
-------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   histogram
   histogram2d
   brazil
   mean_bins
   median_bins
   optimize_nbins_1d
   optimize_nbins_2d
   waverage

Data files and external data
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   read_cdf
   get_omni_data
