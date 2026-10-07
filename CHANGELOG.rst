Changelog
=========

2.5.0 (unreleased)
------------------

This release follows a function-by-function review of ``pyrf`` and ``mms``
against irfu-matlab. It fixes many functions that returned wrong values or
failed on every call, and makes MMS data available from AWS without a local
copy. Several fixes change numerical results: see `Changed behaviour`_ and
`Fixes that change results`_ before updating an analysis.

Requirements
^^^^^^^^^^^^

- Python 3.12 to 3.15. Python 3.10 and 3.11 are no longer supported, following
  PHEP 3.
- NumPy 2.0 or later (the ``<2.4`` upper bound is removed; NumPy 2.5 works).
- ``cdflib`` is no longer used: CDF time conversions use ``pycdfpp`` (now
  ``>=0.9.0``) and NumPy.
- New dependency: ``platformdirs``.

Changed behaviour
^^^^^^^^^^^^^^^^^

These changes can require updating your code.

- ``plot.make_labels``: ``num`` is replaced by ``pref`` and ``suff`` (text
  before and after the letter).
- ``pyrf.c_4_j``: ``b_avg`` is returned in nT, as labelled (it was in T).
- ``pyrf.pvi``: returns the PVI, :math:`|\delta x| / \sqrt{\langle|\delta x|^2\rangle}`
  (Greco et al., 2008), instead of its square.
- ``pyrf.l_shell``: returns L in Earth radii (it was in km).
- ``pyrf.dynamic_press``: takes the density in cm\ :sup:`-3` and the velocity
  in km/s and returns nPa, like the other ``pyrf`` plasma functions (it
  applied SI formulas to the raw values).
- ``pyrf.eb_nrf``: the default flag is ``"a"``, as in irfu-matlab (the old
  default raised an error).
- ``pyrf.find_closest`` returns integer indices, and ``pyrf.corr_deriv``
  returns datetime64 times.
- ``pyrf.wavepolarize_means``: outputs have dimensions ``(time, frequency)``;
  the windows advance by half a window, times are window centres and
  frequencies are those of the FFT bins (as IDL/pyspedas ``wavpol``).
- ``pyrf.psd``: vector and tensor inputs give one spectrum per component, with
  dimensions ``("f", "comp")``.
- ``pyrf.wavelet``: the default highest frequency is the Nyquist frequency (it
  was 100 times higher); a higher ``f_max`` is lowered with a warning.
- ``pyrf.new_xyz``: the output no longer keeps the input's
  ``COORDINATE_SYSTEM``, which made ``cotrans`` treat rotated data as still in
  the old frame. Pass ``coordinate_system`` to set it (``mva`` sets
  ``"lmn"``).
- ``models.igrf`` uses IGRF-14 and the Hapgood (1997) dipole latitude, as
  irfu-matlab. ``pyrf.cotrans`` GSE↔GSM, GSM↔SM and GEO↔MAG change by up to
  0.2°.
- ``models.ion_anisotropy_thresh`` is a wrapper of
  ``pyrf.anisotropy_thresholds``; the proton-cyclotron threshold (and the
  ``plot.ion_brazil_plot_thresh`` curve) is about 6 % lower (coefficient of
  Verscharen et al., 2016).
- ``dispersion.disp_surf_calc``: the ``"(dn_e/n)/(dB/B)"`` key now holds that
  quantity; the one it held before is ``"dn_e/(k E eps0/e)"``. Keys with stray
  spaces still work with a ``FutureWarning``.
- ``dispersion.one_fluid_dispersion``: the branches are sorted by frequency,
  and the default wavenumber range scales with :math:`V_A / \Omega_p` (it was
  fixed).
- ``plot.add_position``: the labels are in Earth radii with an R row, as
  irfu-matlab (``units="km"`` keeps km).
- ``mms.remove_edist_background``: ``n_art`` defaults to ``None`` (the scaling
  factor from the moments file); ``n_art=0`` now removes no photoelectrons.
- ``mms.db_init`` saves the configuration in the user configuration directory
  (``pyrfu.mms.MMS_CFG_PATH``) instead of inside the package, and the SDC
  credentials in the system keyring. The existing configuration is copied the
  first time.
- pyrfu logs to its own ``"pyrfu"`` logger and no longer configures the root
  logger or redirects all warnings at import. Silence it with
  ``logging.getLogger("pyrfu").setLevel(logging.WARNING)``.
- ``mms.get_data`` and ``mms.tokenize``: a variable name made of valid parts
  but not supported (not in ``mms_keys.json``, e.g. ``tsi_fpi_brst_l2``)
  raises ``ValueError`` (it was a bare ``KeyError``).
- ``mms.load_ancillary`` supports ``defatt``, ``defeph`` and ``defq`` and
  raises ``ValueError`` for the predicted products, which were documented but
  raised ``KeyError``; no file in the interval raises ``FileNotFoundError``.
- ``mms.psd_moments``: ``energy_range`` now restricts the integration (it was
  ignored). ``en_channels`` is documented as it works: 0-based
  ``[start, stop)`` indices, unlike irfu-matlab's 1-based ``[min max]``.
- ``mms.get_pitch_angle_dist``: each time step keeps its energy table (the
  energies were the mean of the first two time steps, mislabelling the
  alternating tables of early burst data).
- Invalid inputs raise ``ValueError``, ``TypeError`` or
  ``FileNotFoundError`` instead of ``AssertionError`` in many functions.

Deprecations
^^^^^^^^^^^^

- ``mms.vdf_reduce`` (use ``mms.reduce``) and ``mms.vdf_frame_transformation``
  (to be replaced with the EIS rework) raise a ``FutureWarning``; their known
  issues are listed in their docstrings.
- ``mms.load_brst_segments``: ``data_path`` and ``download`` are deprecated;
  the segments are read from the MMS SDC burst segment service.
- ``dispersion.disp_surf_calc``: keys with stray spaces will be removed in 3.0.

New features
^^^^^^^^^^^^

- MMS data from AWS: ``mms.db_init(default="aws")`` reads the public MMS
  archive on NASA HelioCloud, without AWS credentials. ``source`` arguments
  (``"local"``, ``"sdc"``, ``"aws"``) are added to ``mms.db_get_variable``,
  ``mms.get_feeps_alleyes``, ``mms.get_feeps_omni``, ``mms.hpca_pad`` and
  ``mms.remove_edist_background``.
- ``pyrfu.constants`` with the Earth radius ``R_E`` = 6371.2 km, now used
  throughout the package.
- ``pyrf.histogram2d``: ``scale`` for lin-lin, log-lin, lin-log and log-log
  histograms.
- ``pyrf.iplasma_calc``: ``b_0``, ``n_i``, ``t_e`` and ``t_i`` arguments,
  asking only for the missing ones.
- ``pyrf.anisotropy_thresholds``: 10\ :sup:`-3` and 10\ :sup:`-4` growth
  rates.
- ``mms.average_vdf``: ``n_pts=None`` averages the whole interval.
- ``mms.probe_align_times`` is fully ported, and ``mms.whistler_b2e`` also
  takes spectrograms.
- ``plot.plot_spectr`` and ``pyrf.ts_spectr``: spectrograms with
  time-varying energies; ``plot.colorbar``: ``width``.
- ``mms.psd_moments``: time-dependent ``energy_range``, as an ``(n_t, 2)``
  array or a ``DataArray`` resampled to the distribution times.
- ``mms.get_pitch_angle_dist``: ``meanorsum="sum_weighted"`` (solid-angle
  weighted mean, as irfu-matlab), which was documented but rejected.
- ``dispersion.one_fluid_dispersion``: ``k_vec``; ``mms.lh_wave_analysis``:
  ``vmax``; ``plot.add_position``: ``units``.

Fixes that change results
^^^^^^^^^^^^^^^^^^^^^^^^^

Each of these returned wrong values before.

*Fields, waves and filtering*

- ``pyrf.resample``: averaging windows as irfu-matlab, ``thresh`` (always
  failed), sampling frequency from the median step, and numeric time axes.
- ``pyrf.vht``: error estimate (about 2.4 times too small) and bias from gaps.
- ``pyrf.poynting_flux``: the integral mixed the components.
- ``pyrf.filt``: high-order elliptic filters were unstable; high-pass filters.
- ``pyrf.ebsp``, ``pyrf.compress_cwt`` (NaN averages, off-centre blocks),
  ``pyrf.movmean`` (NaNs spread to later samples), ``pyrf.mean_field``
  (overflow above 65535 samples), ``pyrf.mean`` (dipole sign per sample).

*Multi-spacecraft*

- ``pyrf.c_4_v``: transposed separation matrix (and it raised for every
  input).
- ``pyrf.nanavg_4sc`` averaged NaNs as zeros.
- ``pyrf.cross``, ``pyrf.convert_fac`` and ``pyrf.vht`` paired samples by
  index on offset time grids.

*Coordinates and time*

- ``pyrf.cotrans``: upper-case flags, µs/ms time coordinates (rotations of
  1970), and ``hapgood=False`` sidereal time from UT (0.29°).
- ``pyrf.mva``: ``"MVAR"`` ran the ``"td"`` analysis, and ``"<bn>=0"`` gave
  left-handed frames.
- ``mms.rotate_tensor``: GSE/GSM spin axis converted to radians twice.
- ``mms.dsl2gse``/``mms.dsl2gsm``: spin axes given as vectors are normalised.
- ``pyrf.calc_fs`` (µs/ms times), ``pyrf.unix2datetime64`` (truncation), and
  ``pyrf.ttns2datetime64`` across leap seconds.

*Plasma parameters and models*

- ``pyrf.plasma_beta`` was 10\ :sup:`9` too large.
- ``pyrf.calc_ag``: det P and det G assumed P22 = P33.
- ``pyrf.anisotropy_thresholds`` wrote NaN into the shared beta array.
- ``pyrf.iplasma_calc``: Te default and collision frequency.
- ``pyrf.estimate``: cylinder capacitance about 2 times too low.
- ``models.magnetopause_normal``: distance and normal for z ≠ 0 and x < 0,
  and the bow-shock normal for z = 0.
- ``pyrf.shock_normal`` (model parameters, angle folding) and
  ``pyrf.shock_parameters`` (gyroradius 1000 times too small).
- ``pyrf.get_omni_data`` kept only the rows at the interval endpoints.
- ``dispersion.disp_surf_calc``: group velocity up to 34 times too small and
  ``E_part/E_field``; ``dispersion.one_fluid_dispersion`` lost branches.

*MMS particles*

- ``mms.psd_moments``: Pxz and Pyz kernels, and the energy-table and
  ``partial_moments`` checks.
- ``mms.reduce``: Monte-Carlo speed bias (n, V and T 6-12 % low) and speed
  bin widths.
- ``mms.psd_rebin`` (and ``mms.vdf_to_e64``, ``mms.vdf_projection`` and
  ``mms.reduce`` on burst data with alternating energy tables): when the
  azimuth labels restart between the two samples of a pair (20 % of DIS and
  5 % of DES pairs in 2015), the two energy tables at an azimuth came from
  directions 31° apart (20° in irfu-matlab, which also shifts the wrong way);
  also the time-step overflow (half a sample early), the empty last sample and
  the energy-table order.
- ``mms.vdf_projection``: rebinned angles in degrees used as radians, and phi
  from the wrong times.
- ``mms.hpca_pad``: sample pairing and directions.
- ``mms.remove_edist_background`` (parity 1 model, spin-phase alignment),
  ``mms.remove_imoms_background`` (dynamic pressure units) and
  ``mms.eis_moments`` (P and T 1.5 times too large).
- ``mms.calculate_epsilon``: channels straddling the spacecraft potential.
- ``mms.make_model_vdf``: NaN at every point when the bulk velocity is along
  B (or zero). Channels below the spacecraft potential stay NaN, now
  documented (irfu-matlab gives f(v=0)); they have no weight in
  ``calculate_epsilon`` or moments.
- ``mms.average_vdf`` (window length) and ``mms.vdf_to_e64`` (energy widths).

*Data access*

- ``mms.list_files_aws`` always raised; ``mms.list_files`` returned several
  versions of a file.
- ``mms.get_ts``: dropped the 4th column of every (N, 4) variable (MEC
  quaternions) and gave NaT times for single records.
- ``mms.get_dist``: files without records.
- ``mms.load_ancillary``: the last sample of every DEFEPH file was dropped
  (only DEFATT files end with a ``DATA_STOP`` footer).
- SDC downloads have timeouts, check the HTTP status and no longer leave
  temporary files; ``mms.db_init`` changes apply without restarting Python.

*Plots*

- ``plot.pl_scatter_matrix`` paired the samples by index when the two time
  series have different time grids; they are now paired in time, as for the
  histograms.

Other fixes
^^^^^^^^^^^

- Functions that failed on every call now work: ``pyrf.wavepolarize_means``,
  ``pyrf.waverage``, ``pyrf.corr_deriv``, ``pyrf.find_closest``,
  ``pyrf.remove_repeated_points``, ``pyrf.match_phibe_dir``,
  ``pyrf.match_phibe_v``, ``pyrf.eb_nrf``, ``mms.correct_edp_probe_timing``,
  ``mms.probe_align_times``, ``mms.whistler_b2e``, ``mms.lh_wave_analysis``,
  ``mms.load_brst_segments``, ``mms.feeps_flat_field_corrections``,
  ``plot.mms_pl_config``, ``plot.plot_clines``, ``plot.plot_ang_ang`` and
  ``plot.pl_scatter_matrix`` with ``pdf=True``.
- Functions that failed on common inputs:

  - ``mms.def2psd``, ``mms.dpf2psd``, ``mms.psd2def`` and ``mms.psd2dpf`` on
    one time step, spectra with (time, energy) energies and pitch-angle
    distributions; the species are no longer case-sensitive and accept the
    singular (``"electron"``);
  - ``mms.vdf_elim`` on one time step, and its energy widths are now clipped
    with the energies;
  - ``mms.get_pitch_angle_dist`` and ``mms.vdf_omni`` with 1-D azimuths or one
    time step; the ``vdf_omni`` output for alternating energy tables now keeps
    the units and species, so that the unit conversions work on it;
  - ``lp.photo_current`` with upper-case materials (``"TiN"``) and its listing
    of the materials.

- ``pyrfu.solo`` is available after ``import pyrfu``.
- Memory: ``mms.psd_moments`` (burst speed widths of size n\ :sub:`t`\ :sup:`2`,
  1 GB for 2000 samples), ``mms.make_model_vdf`` and
  ``mms.get_pitch_angle_dist`` no longer tile 4-D arrays.
- The caller's data is no longer modified by ``pyrf.edb``, ``pyrf.vht``,
  ``pyrf.ts_scalar``, ``pyrf.ts_vec_xyz``, ``mms.fft_bandpass``,
  ``mms.estimate_phase_speed``, ``mms.remove_idist_background``,
  ``mms.dist_append``, ``mms.vdf_to_e64`` and
  ``mms.feeps_flat_field_corrections``.
- ``import pyrfu`` no longer imports ``geopack``, which printed
  "Load IGRF coefficients ..." and queried NOAA at every import.
- ``plot.use_pyrfu_style(usetex=True)`` now renders text with LaTeX.

Documentation
^^^^^^^^^^^^^

- Topic-organised API reference (the former ``dev/`` pages redirect to it), a
  new landing page, and a build without warnings.
- New ``pyrf`` examples: time series, four-spacecraft methods, and minimum
  variance and de Hoffmann-Teller analysis.

Known issues
^^^^^^^^^^^^

- ``solo.read_tnr`` fails on every file with data: it calls
  ``scipy.integrate.trapz``, which was removed in SciPy 1.14 (the minimum
  supported version). It will be fixed in 2.6.
