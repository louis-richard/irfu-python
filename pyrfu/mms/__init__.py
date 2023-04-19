#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Louis Richard
from .tokenize import tokenize
from .list_files import list_files
from .get_ts import get_ts
from .get_dist import get_dist
from .db_get_ts import db_get_ts
from .fk_power_spectrum_4sc import fk_power_spectrum_4sc
from .lh_wave_analysis import lh_wave_analysis
from .whistler_b2e import whistler_b2e
from .get_feeps_omni import get_feeps_omni
from .remove_imoms_background import remove_imoms_background
from .remove_idist_background import remove_idist_background
from .remove_edist_background import remove_edist_background
from .psd_moments import psd_moments
from .rotate_tensor import rotate_tensor
from .calculate_epsilon import calculate_epsilon
from .dft_time_shift import dft_time_shift
from .estimate_phase_speed import estimate_phase_speed
from .fft_bandpass import fft_bandpass
from .get_pitch_angle_dist import get_pitch_angle_dist
from .load_brst_segments import load_brst_segments
from .make_model_vdf import make_model_vdf
from .psd_rebin import psd_rebin
from .list_files_ancillary import list_files_ancillary
from .load_ancillary import load_ancillary
from .vdf_omni import vdf_omni
from .spectr_to_dataset import spectr_to_dataset
from .psd2def import psd2def
from .psd2dpf import psd2dpf
from .dpf2psd import dpf2psd
from .def2psd import def2psd
from .copy_files import copy_files
from .copy_files_ancillary import copy_files_ancillary
from .vdf_to_e64 import vdf_to_e64
from .dsl2gse import dsl2gse
from .dsl2gsm import dsl2gsm
from .vdf_projection import vdf_projection
from .vdf_elim import vdf_elim
from .get_data import get_data
from .get_variable import get_variable
from .db_get_variable import db_get_variable
from .make_model_kappa import make_model_kappa
from .vdf_reduce import vdf_reduce, vdf_frame_transformation
from .reduce import reduce

# Hot Plasma Composition Analyser (HPCA)
from .get_hpca_dist import get_hpca_dist
from .hpca_calc_anodes import hpca_calc_anodes
from .hpca_energies import hpca_energies
from .hpca_spin_sum import hpca_spin_sum
from .hpca_pad import hpca_pad

# Fly’s Eye Energetic Particle Spectrometer (FEEPS)
from .feeps_active_eyes import feeps_active_eyes
from .feeps_correct_energies import feeps_correct_energies
from .feeps_energy_table import feeps_energy_table
from .feeps_flat_field_corrections import feeps_flat_field_corrections
from .feeps_omni import feeps_omni
from .feeps_pad import feeps_pad
from .feeps_pad_spinavg import feeps_pad_spinavg
from .feeps_pitch_angles import feeps_pitch_angles
from .feeps_remove_bad_data import feeps_remove_bad_data
from .feeps_remove_sun import feeps_remove_sun
from .feeps_sector_spec import feeps_sector_spec
from .feeps_spin_avg import feeps_spin_avg
from .feeps_split_integral_ch import feeps_split_integral_ch
from .read_feeps_sector_masks_csv import read_feeps_sector_masks_csv
from .get_feeps_alleyes import get_feeps_alleyes
from .db_init import db_init
from .download_data import download_data
from .download_ancillary import download_ancillary
from .correct_edp_probe_timing import correct_edp_probe_timing
from .scpot2ne import scpot2ne
from .probe_align_times import probe_align_times

# Energetic Ion Spectrometer (EIS)
from .eis_ang_ang import eis_ang_ang
from .eis_combine_proton_pad import eis_combine_proton_pad
from .eis_combine_proton_spec import eis_combine_proton_spec

# from .eis_correlation import eis_correlation
from .eis_moments import eis_moments
from .eis_omni import eis_omni
from .eis_pad import eis_pad
from .eis_pad_combine_sc import eis_pad_combine_sc
from .eis_pad_spinavg import eis_pad_spinavg
from .eis_proton_correction import eis_proton_correction
from .eis_skymap import eis_skymap
from .eis_skymap_combine_sc import eis_skymap_combine_sc
from .eis_spec_combine_sc import eis_spec_combine_sc
from .eis_spin_avg import eis_spin_avg
from .get_eis_allt import get_eis_allt
from .eis_combine_proton_skymap import eis_combine_proton_skymap


__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.22"
__status__ = "Prototype"

__all__ = [
    "calculate_epsilon",
    "copy_files",
    "copy_files_ancillary",
    "correct_edp_probe_timing",
    "db_get_ts",
    "db_get_variable",
    "db_init",
    "def2psd",
    "dft_time_shift",
    "download_ancillary",
    "download_data",
    "dpf2psd",
    "dsl2gse",
    "dsl2gsm",
    "eis_ang_ang",
    "eis_combine_proton_pad",
    "eis_combine_proton_skymap",
    "eis_combine_proton_spec",
    "eis_moments",
    "eis_omni",
    "eis_pad",
    "eis_pad_combine_sc",
    "eis_pad_spinavg",
    "eis_proton_correction",
    "eis_skymap",
    "eis_skymap_combine_sc",
    "eis_spec_combine_sc",
    "eis_spin_avg",
    "estimate_phase_speed",
    "feeps_active_eyes",
    "feeps_correct_energies",
    "feeps_energy_table",
    "feeps_flat_field_corrections",
    "feeps_omni",
    "feeps_pad",
    "feeps_pad_spinavg",
    "feeps_pitch_angles",
    "feeps_remove_bad_data",
    "feeps_remove_sun",
    "feeps_sector_spec",
    "feeps_spin_avg",
    "feeps_split_integral_ch",
    "fft_bandpass",
    "fk_power_spectrum_4sc",
    "get_data",
    "get_dist",
    "get_eis_allt",
    "get_feeps_alleyes",
    "get_feeps_omni",
    "get_hpca_dist",
    "get_pitch_angle_dist",
    "get_ts",
    "get_variable",
    "hpca_calc_anodes",
    "hpca_energies",
    "hpca_pad",
    "hpca_spin_sum",
    "lh_wave_analysis",
    "list_files",
    "list_files_ancillary",
    "load_ancillary",
    "load_brst_segments",
    "make_model_kappa",
    "make_model_vdf",
    "probe_align_times",
    "psd2def",
    "psd2dpf",
    "psd_moments",
    "psd_rebin",
    "read_feeps_sector_masks_csv",
    "reduce",
    "remove_edist_background",
    "remove_idist_background",
    "remove_imoms_background",
    "rotate_tensor",
    "scpot2ne",
    "spectr_to_dataset",
    "tokenize",
    "vdf_elim",
    "vdf_frame_transformation",
    "vdf_omni",
    "vdf_projection",
    "vdf_reduce",
    "vdf_to_e64",
    "whistler_b2e",
]
