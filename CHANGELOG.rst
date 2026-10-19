Changelog
=========

2.5.1 (2026-10-19)
------------------

This release follows a review of the FEEPS functions against IDL SPEDAS 6.1
and pyspedas, and ports the methods of the irfu-matlab particle distribution
class (``PDist``). The corrected chain ``get_feeps_alleyes`` →
``feeps_corrections`` → ``feeps_omni`` now matches pyspedas to round-off on
real burst and survey data, and the new FEEPS skymaps can be used as the FPI
distributions. The new ``vdf_*`` and ``pad_*`` functions (frame shift,
rebinning, macroparticles, cuts, ...) are each compared with irfu-matlab on
real FPI data. Most FEEPS products change: see
`Changed behaviour <v2.5.1 changed_>`_ and
`Fixes that change results <v2.5.1 fixes_>`_ before updating an analysis.
Where pyrfu deliberately differs from IDL SPEDAS, pyspedas or irfu-matlab,
the docstring says so.

.. _v2.5.1 changed:

Changed behaviour
^^^^^^^^^^^^^^^^^

These changes can require updating your code.

- ``mms.get_feeps_alleyes``: the samples flagged as not valid for science
  are NaN (new ``quality_flag``, default 3, as the pyspedas omni; ``None``
  keeps all samples; L2 only).
- ``mms.feeps_correct_energies``: the energy dimension of each eye is renamed
  ``energy_top-1``, ``energy_bottom-1``, ..., the name given by
  ``mms.get_feeps_alleyes`` (it was ``energy-top-1``). Code selecting the
  energies of corrected eyes by dimension name must use the new names.
- ``mms.feeps_omni``, ``mms.feeps_pad`` and ``mms.feeps_sector_spec`` raise
  ``ValueError`` on eyes whose integral channel isn't split (16 channels):
  call ``mms.feeps_split_integral_ch`` (or ``mms.feeps_corrections``) first.
  ``feeps_omni`` failed with an unclear broadcast error, and ``feeps_pad``
  and ``feeps_sector_spec`` averaged the integral channel as a spectrum
  channel.
- ``mms.feeps_pad``:

  - burst data use the pitch angles of the CDF (``pitch_angle``), as IDL
    SPEDAS and pyspedas; ``angles_from_bfield=True`` computes them from
    ``b_bcs`` as before;
  - there are floor(180 / ``bin_size``) bins of exactly ``bin_size`` from 0,
    labelled at their centres, as IDL SPEDAS and pyspedas. With a
    ``bin_size`` that doesn't divide 180 (e.g. 25), the bins used to spread
    over 0-180 with labels and windows of ``bin_size``;
  - a 32 keV start is accepted, the invalid inputs raise ``ValueError``, a
    range without energy channel warns (the PAD is NaN), and ``UNITS`` is
    that of the eyes (it was always intensity).

- ``mms.feeps_sector_spec`` returns one row per spin, timed at the spin
  start, and no longer ``spinsectnum`` (the spectra were padded with NaN to
  every sample of ``spinsectnum``).
- ``mms.get_feeps_alleyes``: ``"counts"`` raises ``ValueError`` for L2 data
  (counts are in L1b only), and ``"mask"`` returns the sector mask of each
  sample, ``(time, sectornum)``.

Deprecations
^^^^^^^^^^^^

- ``mms.feeps_pad_spinavg``: ``bin_size`` is ignored (the bins are those of
  the PAD) and will be removed.
- ``mms.vdf_frame_transformation``: the warning points to its replacement,
  ``mms.vdf_shift``; it will be removed in 3.0.

New features
^^^^^^^^^^^^

- The methods of the irfu-matlab ``PDist`` class, each compared with
  irfu-matlab on real FPI data (the differences are irfu-matlab bugs, or
  deliberate and documented in the docstrings):

  - ``mms.vdf_convert`` (``PDist.convertto``): phase-space densities
    (s\ :sup:`3` cm\ :sup:`-6`, m\ :sup:`-6` or km\ :sup:`-6`),
    differential energy and particle fluxes;
  - ``mms.vdf_velocity`` (``PDist.v``), ``mms.vdf_solid_angle``
    (``solidangle``) and ``mms.vdf_d3v`` (``d3v``): the velocities, solid
    angles and velocity-space volumes of the bins, from the ``delta_*``
    attributes when they are given;
  - ``mms.vdf_partial_density`` (``dn`` and ``dj``) and ``mms.vdf_flux``
    (``flux``);
  - ``mms.vdf_mask`` (``mask``) and ``mms.vdf_find_low_counts``
    (``find_low_counts``);
  - ``mms.pad_palim`` (``palim``) and ``mms.pad_diffusion``
    (``pitchangle_diffusion``, with the derivatives in radians);
  - ``mms.vdf_shift`` (``shift``): the distribution in a moving frame and in
    rotated coordinates;
  - ``mms.vdf_rebin`` (``rebin``): the distribution on new spherical or
    Cartesian bins;
  - ``mms.vdf_macroparticles`` (``macroparticles``), as one Dataset;
  - ``mms.vdf_cuts`` (``cuts``): cuts along and perpendicular to B, with
    their moments.

  The Monte-Carlo particles of ``vdf_shift``, ``vdf_rebin`` and
  ``vdf_macroparticles`` are uniform in the volume of each bin (in
  v\ :sup:`3` and cos θ); irfu-matlab draws them uniformly in energy and
  polar angle, which biases them towards the low speeds and the poles.

- ``mms.vdf_omni``: ``v_frame`` gives the omni-directional distribution in a
  moving frame, as irfu-matlab ``get_omni_dist_framedep``, e.g. in the normal
  incidence frame of a shock.
- ``plot.plot_pad_polar`` (irfu-matlab ``PDist.plot_pad_polar``): a
  pitch-angle distribution in the (v\ :sub:`∥`, v\ :sub:`⊥`) half plane,
  drawn from its bins or resampled with Monte-Carlo particles.
- ``mms.psd_moments``: ``pitch_angle_range`` with ``b_xyz`` gives the moments
  of the particles in a range of pitch angles, the bins selected by their
  centre (as the partial moments of the irfu-matlab examples) or weighted by
  the fraction of their solid angle in the range
  (``pitch_angle_weight="fraction"``, for wide bins).
- ``mms.get_dist``: the half widths of the accumulations, ``delta_time_minus``
  and ``delta_time_plus`` (s, per time), as irfu-matlab; ``pyrf.dist_append``,
  ``mms.vdf_to_e64``, ``pyrf.average_vdf`` and ``mms.vdf_shift`` carry them.
- ``mms.get_eis_allt``: ``source`` (``"default"``, ``"local"``, ``"sdc"`` or
  ``"aws"``); each file is read once.
- ``mms.feeps_skymap``: FEEPS skymaps, one per spin (time, energy, phi,
  theta), in DMPA and in relativistic phase-space density (s\ :sup:`3`
  cm\ :sup:`-6`), as the FPI distributions, for ``mms.psd_moments``,
  ``mms.reduce`` and ``mms.vdf_projection``. The look directions are despun with the spin
  sectors (a constant per spacecraft, measured on 2015-2024 FGM data), or with
  FGM (``b_bcs`` and ``b_dmpa``). The 4 polar rings of the eyes give coarse
  partial moments above about 30 keV; ``coverage`` gives the fraction of the
  sphere with data, and ``fill_empty`` optionally fills the bins without
  data.
- ``mms.reduce`` uses the angular widths of the skymap (``delta_theta_*`` and
  ``delta_phi_*``) when they are given, as for the FEEPS skymaps, whose 4
  polar bins have different widths; FPI and EIS skymaps are unchanged.
- ``mms.dpf2psd`` and ``mms.psd2dpf``: ``relativistic=True`` converts to
  and from the relativistic phase-space density, f = γ\ :sup:`3` m J /
  v\ :sup:`2` with the relativistic speed v, whose integral over the
  velocities is the density. For electrons, it is larger than the
  non-relativistic f = m J / v\ :sup:`2` (v\ :sup:`2` = 2E/m, the default, as
  FPI) by 4.5 % at 5 keV, 52 % at 50 keV and a factor 2.2 at 100 keV. The
  FEEPS skymaps use it.
- ``mms.psd_moments``: ``relativistic=True`` uses the relativistic speeds,
  momentum (pressure) and kinetic energy (heat flux), for the relativistic
  phase-space density; the default is the ``relativistic`` attribute of the
  distribution (True for the FEEPS skymaps), else False, as before. The
  density is then the integral of the intensity over the energy and the solid
  angle; the non-relativistic moments underestimate the FEEPS electron density
  by about 10 % (more for harder spectra).
- ``mms.get_gyrophase_dist``: gyrophase distributions of a skymap (FPI, EIS,
  FEEPS), optionally in a range of pitch angles. The gyrophase is measured
  right-handed about B from the component of a reference direction
  perpendicular to B (default [1, 0, 0], as ``pyrf.convert_fac``), as Li et
  al. (2022).
- ``mms.get_feeps_alleyes``: the lower and upper bounds of the energy
  channels of each eye, from the CDF, as coordinates along its energies
  (``energy_lower_bound_top-1``, ...).
- ``mms.psd_moments``: ``angle_integration="exact"`` integrates the angular
  kernels over each bin instead of taking them at the bin centre (the
  default, as irfu-matlab). On the FPI grid the moments change by less than
  1 %; on the wide FEEPS bins the midpoint rule underestimates the density by
  8 to 11 %. The FEEPS skymaps use "exact" by default (``angle_integration``
  attribute).
- ``mms.feeps_pad``: ``angles_from_bfield`` (see above).
- ``mms.feeps_remove_sunlit_sectors``: ``i_en``, the energy channel used to
  find the sunlit sectors (default 3, as before).
- The FEEPS sun-contamination masks of 2025-08-14 and 2025-12-26, and the
  updated MMS4 masks of 2025-02-16.

.. _v2.5.1 fixes:

Fixes that change results
^^^^^^^^^^^^^^^^^^^^^^^^^

Each of these returned wrong values before.

- ``mms.psd_moments`` took every distribution as s\ :sup:`3` cm\ :sup:`-6`
  (FPI): the moments of a distribution in s\ :sup:`3` m\ :sup:`-6`
  (``mms.dpf2psd``, ``mms.eis_skymap``) were 10\ :sup:`12` too large, and in
  s\ :sup:`3` km\ :sup:`-6` (``mms.make_model_vdf``) 10\ :sup:`30`. It now
  reads ``UNITS`` (s\ :sup:`3` cm\ :sup:`-6` if missing) and converts the
  differential energy and particle fluxes with ``mms.vdf_convert``, as
  irfu-matlab. The FPI moments are unchanged.
- ``mms.vdf_omni`` weighted the angles by sin θ, which is exact only on
  uniform polar grids: the weights are the solid angles of the bins. The FEEPS
  skymaps were 4 % off; the FPI omni is unchanged.
- ``mms.vdf_to_e64`` merged a distribution with a single energy table into 64
  duplicated channels at half the time resolution; it is returned unchanged,
  as irfu-matlab ``PDist.e64``.
- ``mms.feeps_remove_bad_data``:

  - used the 2017 bad-eye table for all dates. It now takes the table
    nearest in time, as pyspedas (data from 2018-04-01 12:00 UTC on use the
    2018 table);
  - never removed the ion eyes (6-8) or their bad channels, nor the bad
    lowest channels of eyes without energy calibration, as IDL SPEDAS.

- ``mms.read_feeps_sector_masks_csv`` (and ``mms.feeps_remove_sun``) used the
  2018-10-05 masks for all later data; the mask nearest in time is used, as
  IDL SPEDAS and pyspedas.
- ``mms.feeps_flat_field_corrections``: the eyes with a gain of 0 (MMS2
  top-7 and bottom-7, MMS4 top-7) are NaN; they were zeros, which pulled the
  ion omni down by about 20 %.
- ``mms.feeps_omni``: the channels of an eye whose energy is more than 10 %
  from the omni energy are left out, as IDL SPEDAS and pyspedas.
- ``mms.feeps_spin_avg`` and ``mms.feeps_pad_spinavg``: each spin is
  averaged from its first sample up to the first sample of the next spin
  (the windows were shifted by one sample, as in IDL SPEDAS and pyspedas),
  the partial spins at the ends are kept, a spin without data is NaN (the
  last rows were zeros), and the attributes are kept. ``feeps_spin_avg`` with
  a ``DataArray`` of spin sectors (as ``mms.get_feeps_omni(spin_avg=True)``)
  returned the samples unaveraged; ``feeps_pad_spinavg`` averages on the PAD
  bins (it interpolated onto the bin edges) and failed with NumPy spin
  sectors.
- ``mms.feeps_avg_4sc`` of spin-averaged spectra: the zero rows of the spin
  average counted as data, and ``combined_energies`` raised ``KeyError``.
- ``mms.feeps_sector_spec``: each spin is stamped at its own start (it was
  one row late, with a zero first row), and unsampled sectors are NaN (they
  were 0).
- ``mms.feeps_pitch_angles``: the survey pitch angles are interpolated onto
  the FEEPS times, as pyspedas (B was averaged over the 2.5 s accumulation,
  during which it turns by about 44° in body coordinates); B exactly along an
  eye gives 0° or 180° (it was NaN).
- ``mms.feeps_remove_sunlit_sectors``: the eye-sector pairs are averaged, as
  documented (they were summed), and the pairs without data are left out of
  the quantile. In survey data, the 56 empty sectors of each eye counted as
  zeros, so about 9 % of a survey day was masked at ``outliers_q=0.99``
  (1-3 % now).

Other fixes
^^^^^^^^^^^

- ``solo.read_tnr`` failed on every file with data: it called
  ``scipy.integrate.trapz``, removed in SciPy 1.14 (the known issue of 2.5.0).
- ``mms.reduce`` raised ``KeyError`` on skymaps without
  ``delta_energy_plus``; the default Cartesian grid then extends to the upper
  edge of the speed bins inferred from the channels.
- ``mms.get_eis_allt`` could not read any real EIS file: the variable names
  were wrong since 2020 and for the files of minor version 0, and the energy
  deltas failed in ``assign_coords``. Old (non-reprocessed) files, a species
  missing from the file and an energy table changing in the time interval
  raise ``ValueError``.
- ``mms.psd_moments`` refused the distributions whose ``FIELDNAM`` doesn't
  name the burst or fast rate (e.g. survey skymaps); the rate was only
  logged.
- ``mms.feeps_remove_bad_data``, ``mms.feeps_remove_sun`` and
  ``mms.feeps_pad`` no longer change the caller's data; ``feeps_pad`` works on
  integer counts.
- ``mms.feeps_split_integral_ch`` cut the last column of ``pitch_angle`` (the
  last bottom eye) and dropped the variables other than the eyes and
  ``spinsectnum``; only the eyes are split now.
- ``mms.get_feeps_alleyes`` failed for ``"mask"`` and read the L2 dataset for
  every level.
- ``mms.feeps_active_eyes`` accepts ``datetime.datetime`` and
  ``numpy.datetime64`` time intervals.
- ``mms.feeps_avg_4sc`` accepts tuples and no longer warns on all-NaN times.
- ``mms.feeps_correct_energies`` no longer shares its attributes with the
  input, and raises a clear ``ValueError`` on already split eyes.
- The sun-mask files of 2024-02-02 with a byte order mark are read.
- ``mms.feeps_remove_sunlit_sectors`` and ``mms.feeps_pad`` are 50 to 300
  times faster on long intervals.

Documentation
^^^^^^^^^^^^^

- ``mms.get_feeps_omni`` is documented as a quick look without the energy
  and flat-field corrections; the docstring gives the corrected chain.
- ``mms.feeps_remove_sun``: survey samples, which average 8 sectors, are
  masked only by the sector they are labelled with, as IDL SPEDAS and
  pyspedas.
- The FEEPS examples are updated for these changes.
- New example: ion gyrophase bunching by EMIC waves, reproducing Fig. 1 of
  Li et al. (2022) with ``mms.get_gyrophase_dist``.
- The IP shock example follows irfu-matlab ``Example_MMS_IPshock``: data from
  AWS, the omni-directional ion distribution in the normal incidence frame
  (``mms.vdf_omni`` with ``v_frame``) and the event of 2023-12-17.
- The README, the docs and ``CITATION.cff`` describe pyrfu on its own;
  IRFU-MATLAB stays credited as its origin.

Development
^^^^^^^^^^^

- The pre-commit hook that ran ``pre-commit autoupdate`` on every commit is
  removed (it failed the commits and updated the linters without review);
  black and pylint are updated.
- The CodeQL workflow, which never ran, and its badge are removed.
- The docs build again without warnings (``sphinx-build -W``).

2.5.0 (2026-10-07)
------------------

This release follows a function-by-function review of ``pyrf`` and ``mms``
against irfu-matlab. It fixes many functions that returned wrong values or
failed on every call, and makes MMS data available from AWS without a local
copy. Several fixes change numerical results: see
`Changed behaviour <v2.5 changed_>`_ and
`Fixes that change results <v2.5 fixes_>`_ before updating an analysis.

Requirements
^^^^^^^^^^^^

- Python 3.12 to 3.15. Python 3.10 and 3.11 are no longer supported, following
  PHEP 3.
- NumPy 2.0 or later (the ``<2.4`` upper bound is removed; NumPy 2.5 works).
- ``cdflib`` is no longer used: CDF time conversions use ``pycdfpp`` (now
  ``>=0.9.0``) and NumPy.
- New dependency: ``platformdirs``.

.. _v2.5 changed:

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
  ``"lmn"``). ``pyrf.eb_nrf`` also sets ``"lmn"``.
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
  first time. Only the settings given change (each call used to reset the
  others to their defaults, e.g. the SDC rights to public); ``reset=True``
  restores the defaults. The credentials are stored only when both
  ``sdc_username`` and ``sdc_password`` are given (a placeholder
  "username"/"password" was stored otherwise).
- ``solo.db_init`` and ``maven.db_init`` also save their data path in the user
  configuration directory (``solo_config.json``, ``maven_config.json``)
  instead of the package ``config.json``. The package files no longer hold a
  local path: set yours again with ``db_init`` after updating.
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
- ``mms.get_data``: ``r_gse_mec_brst_l2``; the docstring lists all the keys.
- ``dispersion.one_fluid_dispersion``: ``k_vec``; ``mms.lh_wave_analysis``:
  ``vmax``; ``plot.add_position``: ``units``.

.. _v2.5 fixes:

Fixes that change results
^^^^^^^^^^^^^^^^^^^^^^^^^

Each of these returned wrong values before.

*Fields, waves and filtering*

- ``pyrf.resample``: averaging windows as irfu-matlab, ``thresh`` (always
  failed), sampling frequency from the median step, and numeric time axes.
- ``pyrf.vht``: error estimate (about 2.4 times too small) and bias from gaps.
- ``pyrf.poynting_flux``: the integral mixed the components.
- ``pyrf.filt``: high-order elliptic filters were unstable; high-pass filters.
- ``pyrf.wavelet``: with linear spacing, the frequencies ignored the requested
  range (``f=[1, 10]`` gave 1 to 64 Hz); they are now ``df``, ``2 df``, ...
  up to the highest frequency of ``f``, as irfu-matlab.
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
- ``lp.thermal_current``: ``"Sphere"`` (not lower case) gave the cylinder
  current, up to 25 % different.
- ``models.magnetopause_normal``: distance and normal for z ≠ 0 and x < 0,
  and the bow-shock normal for z = 0.
- ``pyrf.shock_normal`` (model parameters, angle folding) and
  ``pyrf.shock_parameters`` (gyroradius 1000 times too small).
- ``pyrf.get_omni_data`` kept only the rows at the interval endpoints.
- ``dispersion.disp_surf_calc``: group velocity up to 34 times too small and
  ``E_part/E_field``; ``dispersion.one_fluid_dispersion`` lost branches.

*MMS particles*

- ``mms.psd_moments``: the speed widths of the energy channels now subtract
  the spacecraft potential from the channel edges, as irfu-matlab: electron
  densities increase by 4-20 % for large potentials on FPI data (cold
  electrons were 10-30 % low at 20 eV and 5-10 V); ion densities stay within
  0.5 % of the FPI moments. Also the Pxz and Pyz kernels, the energy-table and
  ``partial_moments`` checks, and the ``delta_phi``/``delta_theta``
  attributes, used in degrees as radians (nothing in pyrfu sets them yet).
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
  ``mms.remove_imoms_background`` (dynamic pressure units; samples with a
  negative corrected density are NaN) and
  ``mms.eis_moments`` (P and T 1.5 times too large).
- ``mms.calculate_epsilon``: channels straddling the spacecraft potential.
- ``mms.make_model_vdf``: NaN at every point when the bulk velocity is along
  B (or zero), or when B is exactly along x (through ``mms.rotate_tensor``,
  which also gave NaN for a ``"rot"`` vector along y); ``mms.calculate_epsilon``
  was then 0. Channels below the spacecraft potential stay NaN, now
  documented (irfu-matlab gives f(v=0)); they have no weight in
  ``calculate_epsilon`` or moments.
- ``mms.average_vdf`` (window length) and ``mms.vdf_to_e64`` (energy widths).
- ``pyrf.int_sph_dist``: NaN values gave NaN bins (an all-zero projection with
  ``weight="lin"`` or ``"log"``), and the log weighting left values below
  10\ :sup:`-16` (SI units) without Monte-Carlo particles.

*Data access*

- ``mms.get_data``: ``vel_gsm_mec_srvy_l2`` and ``vel_gsm_mec_brst_l2``
  returned the GSE velocity.
- ``mms.list_files_aws`` always raised; ``mms.list_files`` returned several
  versions of a file.
- ``mms.get_ts``: dropped the 4th column of every (N, 4) variable (MEC
  quaternions) and gave NaT times for single records.
- ``mms.get_dist``: files without records.
- ``mms.load_ancillary``: the last sample of every DEFEPH file was dropped
  (only DEFATT files end with a ``DATA_STOP`` footer).
- ``mms.get_data`` failed for the whole interval when a file had no records
  for a variable with a time-varying table (FPI omnidirectional spectra), or
  had exactly 4 records.
- The SDC session sent the SITL username and password with every public
  request; they are now only read and sent for the SITL access.
- SDC downloads have timeouts, check the HTTP status and no longer leave
  temporary files; the SDC login happens once per session and retries on
  rate limiting and server errors; ``mms.db_init`` changes apply without
  restarting Python.

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
  ``pyrf.brazil``, ``plot.mms_pl_config``, ``plot.plot_clines``,
  ``plot.plot_ang_ang`` and ``plot.pl_scatter_matrix`` with ``pdf=True``.
- Functions that failed on common inputs:

  - ``mms.psd_moments`` on every ``mms.get_dist`` output (also after
    ``pyrf.time_clip`` or ``mms.dist_append``), on energy tables with fewer
    than 32 channels (``mms.vdf_elim`` output), and without
    ``delta_energy`` attributes;
  - ``pyrf.histogram2d`` with a NumPy ``bins=[nx, ny]``, and constant data
    (zero-width bins); it raises a clear ``ValueError`` when no sample is
    left;
  - ``pyrf.shock_normal`` with lists, ``pyrf.nanavg_4sc`` with skymaps
    (``Dataset``), ``pyrf.filt`` with NumPy scalar cut-offs and orders, and
    ``pyrf.gse2gsm`` with upper-case flags;
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
- The caller's data is no longer modified by ``pyrf.cotrans`` (a same-frame
  transformation returned the input), ``pyrf.shock_normal``, ``pyrf.edb``,
  ``pyrf.vht``,
  ``pyrf.ts_scalar``, ``pyrf.ts_vec_xyz``, ``mms.fft_bandpass``,
  ``mms.estimate_phase_speed``, ``mms.remove_idist_background``,
  ``mms.dist_append``, ``mms.vdf_to_e64`` and
  ``mms.feeps_flat_field_corrections``.
- ``import pyrfu`` no longer imports ``geopack``, which printed
  "Load IGRF coefficients ..." and queried NOAA at every import.
- ``plot.use_pyrfu_style(usetex=True)`` now renders text with LaTeX.
- ``pyrf.e_vxb``: the units are "mV/m" (they were labelled "mV/s").
- The wheel contains only ``pyrfu`` (it also shipped the documentation
  sources as a top-level ``docs`` package).

Documentation
^^^^^^^^^^^^^

- Topic-organised API reference (the former ``dev/`` pages redirect to it), a
  new landing page, and a build without warnings (now checked in CI).
- New examples: MMS data access (local, SDC and AWS, and downloads), time
  series, four-spacecraft methods, and minimum variance and de
  Hoffmann-Teller analysis.
- Explicit titles and a description for every example. The images of the
  HPCA and FEEPS four-spacecraft examples show again, and the polarization
  analysis example runs again.

Development
^^^^^^^^^^^

- The test suite runs offline (about 40 s): the SDC tests use a fake
  session, and one real query of the public SDC runs with
  ``PYRFU_NETWORK_TESTS=1``.

Known issues
^^^^^^^^^^^^

- ``solo.read_tnr`` fails on every file with data: it calls
  ``scipy.integrate.trapz``, which was removed in SciPy 1.14 (the minimum
  supported version). Fixed in 2.5.1.
