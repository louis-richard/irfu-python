#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Louis Richard
from .read_cdf import read_cdf
from .ts_time import ts_time
from .ts_scalar import ts_scalar
from .ts_vec_xyz import ts_vec_xyz
from .ts_tensor_xyz import ts_tensor_xyz
from .ts_skymap import ts_skymap
from .ts_append import ts_append
from .dist_append import dist_append
from .start import start
from .end import end
from .iso2unix import iso2unix
from .extend_tint import extend_tint
from .dot import dot
from .cross import cross
from .trace import trace
from .norm import norm
from .normalize import normalize
from .calc_dt import calc_dt
from .calc_fs import calc_fs
from .avg_4sc import avg_4sc
from .gradient import gradient
from .integrate import integrate
from .time_clip import time_clip
from .resample import resample
from .t_eval import t_eval
from .filt import filt
from .medfilt import medfilt
from .movmean import movmean
from .wavelet import wavelet
from .compress_cwt import compress_cwt
from .mva import mva
from .vht import vht
from .new_xyz import new_xyz
from .dec_par_perp import dec_par_perp
from .convert_fac import convert_fac
from .edb import edb
from .e_vxb import e_vxb
from .plasma_calc import plasma_calc
from .dynamic_press import dynamic_press
from .pres_anis import pres_anis
from .calc_sqrtq import calc_sqrtq
from .median_bins import median_bins
from .mean_bins import mean_bins
from .histogram import histogram
from .histogram2d import histogram2d
from .wavepolarize_means import wavepolarize_means
from .psd import psd
from .c_4_k import c_4_k
from .c_4_grad import c_4_grad
from .c_4_j import c_4_j
from .c_4_v import c_4_v
from .ebsp import ebsp
from .date_str import date_str
from .find_closest import find_closest
from .cart2sph_ts import cart2sph_ts
from .corr_deriv import corr_deriv
from .eb_nrf import eb_nrf
from .lowpass import lowpass
from .mean import mean
from .poynting_flux import poynting_flux
from .remove_repeated_points import remove_repeated_points
from .solid_angle import solid_angle
from .wave_fft import wave_fft
from .pvi import pvi
from .pvi_4sc import pvi_4sc
from .pid_4sc import pid_4sc
from .calc_dng import calc_dng
from .calc_ag import calc_ag
from .calc_agyro import calc_agyro
from .datetime2iso8601 import datetime2iso8601
from .cotrans import cotrans
from .sph2cart import sph2cart
from .gse2gsm import gse2gsm
from .mean_field import mean_field
from .waverage import waverage
from .l_shell import l_shell
from .optimize_nbins_1d import optimize_nbins_1d
from .optimize_nbins_2d import optimize_nbins_2d
from .magnetosphere import magnetosphere
from .iso86012timevec import iso86012timevec
from .timevec2iso8601 import timevec2iso8601
from .iso86012datetime64 import iso86012datetime64
from .datetime642iso8601 import datetime642iso8601
from .unix2datetime64 import unix2datetime64
from .datetime642unix import datetime642unix
from .ttns2datetime64 import ttns2datetime64
from .datetime642ttns import datetime642ttns
from .iso86012datetime import iso86012datetime
from .cdfepoch2datetime64 import cdfepoch2datetime64
from .cart2sph import cart2sph
from .iplasma_calc import iplasma_calc
from .match_phibe_dir import match_phibe_dir
from .match_phibe_v import match_phibe_v
from .get_omni_data import get_omni_data
from .estimate import estimate
from .struct_func import struct_func
from .autocorr import autocorr
from .increments import increments
from .int_sph_dist import int_sph_dist
from .shock_normal import shock_normal
from .shock_parameters import shock_parameters

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"

__all__ = [
    "autocorr",
    "average_vdf",
    "avg_4sc",
    "c_4_grad",
    "c_4_j",
    "c_4_k",
    "c_4_v",
    "calc_ag",
    "calc_agyro",
    "calc_dng",
    "calc_dt",
    "calc_fs",
    "calc_sqrtq",
    "cart2sph",
    "cart2sph_ts",
    "cdfepoch2datetime64",
    "compress_cwt",
    "convert_fac",
    "corr_deriv",
    "cotrans",
    "cross",
    "date_str",
    "datetime2iso8601",
    "datetime642iso8601",
    "datetime642ttns",
    "datetime642unix",
    "dec_par_perp",
    "dist_append",
    "dot",
    "dynamic_press",
    "e_vxb",
    "eb_nrf",
    "ebsp",
    "edb",
    "end",
    "estimate",
    "extend_tint",
    "filt",
    "find_closest",
    "get_omni_data",
    "gradient",
    "gse2gsm",
    "histogram",
    "histogram2d",
    "increments",
    "int_sph_dist",
    "integrate",
    "iplasma_calc",
    "iso2unix",
    "iso86012datetime",
    "iso86012datetime64",
    "iso86012timevec",
    "l_shell",
    "lowpass",
    "magnetosphere",
    "match_phibe_dir",
    "match_phibe_v",
    "mean",
    "mean_bins",
    "mean_field",
    "medfilt",
    "median_bins",
    "movmean",
    "mva",
    "new_xyz",
    "norm",
    "normalize",
    "optimize_nbins_1d",
    "optimize_nbins_2d",
    "pid_4sc",
    "plasma_beta",
    "plasma_calc",
    "poynting_flux",
    "pres_anis",
    "psd",
    "pvi",
    "pvi_4sc",
    "read_cdf",
    "remove_repeated_points",
    "resample",
    "shock_normal",
    "shock_parameters",
    "solid_angle",
    "sph2cart",
    "st_diff",
    "start",
    "struct_func",
    "t_eval",
    "time_clip",
    "timevec2iso8601",
    "trace",
    "ts_append",
    "ts_scalar",
    "ts_skymap",
    "ts_tensor_xyz",
    "ts_time",
    "ts_vec_xyz",
    "ttns2datetime64",
    "unix2datetime64",
    "vht",
    "wave_fft",
    "wavelet",
    "wavepolarize_means",
    "waverage",
]
