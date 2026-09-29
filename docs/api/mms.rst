pyrfu.mms
=========

.. module:: pyrfu.mms

.. currentmodule:: pyrfu.mms

Routines specific to the Magnetospheric Multiscale (MMS) mission: data access,
field and wave analysis, FPI particle distributions and moments, and the
energetic particle instruments (EIS, FEEPS) and HPCA.

.. code-block:: python

    from pyrfu import mms

    mms.db_init(default="local", local="/Volumes/mms")

    tint = ["2019-09-14T07:54:00.000", "2019-09-14T08:11:00.000"]
    b_gse = mms.get_data("b_gse_fgm_srvy_l2", tint, 1)

Data access configuration
-------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   db_init
   db_get_ts
   db_get_variable

Finding and downloading files
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   list_files
   list_files_sdc
   list_files_aws
   list_files_ancillary
   list_files_ancillary_sdc
   download_data
   download_ancillary
   copy_files
   copy_files_ancillary
   load_brst_segments

Loading data
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_data
   get_ts
   get_variable
   get_dist
   get_pitch_angle_dist
   load_ancillary
   tokenize

Fields, coordinates and spacecraft potential
--------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dsl2gse
   dsl2gsm
   rotate_tensor
   correct_edp_probe_timing
   dft_time_shift
   fft_bandpass
   scpot2ne
   probe_align_times

Waves
-----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fk_power_spectrum_4sc
   lh_wave_analysis
   whistler_b2e
   estimate_phase_speed

FPI particle distributions
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   vdf_elim
   vdf_omni
   vdf_to_e64
   psd_rebin
   vdf_frame_transformation
   vdf_projection
   reduce
   vdf_reduce
   make_model_vdf
   make_model_kappa
   calculate_epsilon

Moments and background removal
------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   psd_moments
   remove_edist_background
   remove_idist_background
   remove_imoms_background

Unit conversions
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   def2psd
   dpf2psd
   psd2def
   psd2dpf
   spectr_to_dataset

Energetic Ion Spectrometer (EIS)
--------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_eis_allt
   eis_omni
   eis_spin_avg
   eis_moments
   eis_pad
   eis_pad_spinavg
   eis_pad_combine_sc
   eis_ang_ang
   eis_skymap
   eis_skymap_combine_sc
   eis_spec_combine_sc
   eis_combine_proton_spec
   eis_combine_proton_pad
   eis_combine_proton_skymap
   eis_proton_correction

Fly's Eye Energetic Particle Spectrometer (FEEPS)
-------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_feeps_alleyes
   get_feeps_omni
   feeps_active_eyes
   feeps_energy_table
   feeps_corrections
   feeps_correct_energies
   feeps_flat_field_corrections
   feeps_remove_bad_data
   feeps_remove_sun
   feeps_remove_sunlit_sectors
   read_feeps_sector_masks_csv
   feeps_split_integral_ch
   feeps_omni
   feeps_spin_avg
   feeps_avg_4sc
   feeps_pitch_angles
   feeps_pad
   feeps_pad_spinavg
   feeps_sector_spec

Hot Plasma Composition Analyzer (HPCA)
--------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_hpca_dist
   hpca_energies
   hpca_calc_anodes
   hpca_spin_sum
   hpca_pad
