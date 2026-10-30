Changelog
=========

2.5.8 (2026-10-30)
------------------

This release fixes the MMS EIS routines, checked against pySPEDAS, the IDL
SPEDAS source and FPI-DIS: the pitch angle distributions keep the zero
fluxes, the spin averages keep the complete spins only, the spacecraft are
combined at the same look directions and times, the PHxTOF and ExTOF
channels are combined as IDL SPEDAS 6.1, the angle-angle distributions bin
every sample, and the partial moments integrate over the channel widths.
The conversions between phase-space density and fluxes use the proton
mass, as the FPI products (1.46 % for protons and electrons), and He2+ for
the alphas. The FEEPS spin averages, skymaps and sector spectrograms also keep
the complete spins only.

Changed behaviour
^^^^^^^^^^^^^^^^^

- ``mms.eis_ang_ang`` gives the look directions in DBCS by default
  (``frame="dbcs"``) and raises ``ValueError`` without ``defatt``; it gave
  them in GSE without ``defatt``. Use ``frame="gse"`` for GSE.
  ``mms.eis_combine_proton_skymap`` passes ``defatt`` and ``frame``.
- Spin averages keep the complete spins only (the partial first and last
  spins and the spins cut by data gaps were kept), each at the time of its
  first sample, and raise ``ValueError`` without a complete spin:
  ``mms.eis_spin_avg``, ``mms.eis_pad_spinavg``, ``mms.eis_ang_ang``,
  ``mms.feeps_spin_avg``, ``mms.feeps_pad_spinavg``, ``mms.feeps_skymap``,
  ``mms.feeps_sector_spec`` and ``mms.get_feeps_omni(spin_avg=True)``.
  ``mms.eis_spin_avg`` was at the 5th of the 8 survey samples of a spin, and
  ``mms.eis_pad_spinavg`` one spin early.
- ``mms.eis_pad`` raises ``ValueError`` without ``vec`` (the pitch angles
  were taken from X-GSE), for a ``vec`` that is neither a time series nor a
  direction, when no channel is in ``energy``, and for a ``pa_width`` that
  doesn't divide 180 (the bins had gaps).
- ``mms.eis_combine_proton_spec``, ``mms.eis_moments`` and
  ``mms.eis_proton_correction`` take fluxes only, as IDL SPEDAS: they raise
  ``ValueError`` for counts and count rates (``UNITS`` of the CDF), and
  ``mms.eis_combine_proton_spec`` for PHxTOF and ExTOF in different units.
  Data without ``UNITS`` are not checked.
- ``ValueError`` for invalid inputs: ``mms.eis_omni`` and
  ``mms.eis_spec_combine_sc`` (unknown method, was ``AssertionError`` in
  ``mms.eis_omni``), ``mms.eis_pad_spinavg`` (spin numbers not on the times
  of the PAD: a PAD of a sub-interval was misaligned without warning),
  ``mms.eis_combine_proton_spec`` (different telescopes, cross-over channels
  that don't pair), ``mms.eis_pad_combine_sc`` (different pitch angle bins),
  ``mms.eis_skymap_combine_sc`` (different units or species, unknown
  method) and ``mms.eis_moments`` (no energy widths).
  ``mms.eis_proton_correction`` raises ``TypeError`` (was
  ``AssertionError``) for an input that isn't xarray.

New features
^^^^^^^^^^^^

- ``mms.eis_ang_ang`` and ``mms.eis_combine_proton_skymap``: ``frame``,
  ``"dbcs"`` or ``"gse"``.
- ``mms.eis_pad`` accepts a direction as a NumPy array (``TypeError``).
- ``mms.eis_moments`` accepts ``"helium"``.
- Attributes: ``mms.eis_omni`` keeps the energy widths and the species, and
  the spin numbers and sectors as coordinates; ``mms.eis_pad`` gives the
  units (``UNITS``, ``LABLAXIS``), species, energy widths of the channels
  used and ``pa_width`` (it had none), and ``mms.eis_pad_spinavg`` keeps
  them; the outputs of ``mms.eis_moments`` have their units.

Fixes that change results
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``mms.psd2def``, ``mms.psd2dpf``, ``mms.def2psd``, ``mms.dpf2psd`` and
  ``mms.vdf_convert`` use the proton mass, as the FPI products: the
  constant m_p^2 / (2 e^2) is 0.54494. It was 0.53707, with the atomic mass
  unit as irfu-matlab: for protons and electrons, the fluxes were 1.46 %
  too large and the phase-space densities 1.46 % too small (FPI-DIS
  ``defi`` / ``psd2def(vdf_omni(pdi))`` was 0.985, now 0.9993). The alphas
  (``"alpha"``, ``"alphas"``, ``"helium"``) are He2+ ions, 3.9726 proton
  masses (they were 4): their conversions change by 0.08 % (4 atomic mass
  units were close to He2+), and their speeds in the ``vdf_*`` routines by
  0.35 %.
- ``mms.eis_pad`` keeps the zero fluxes, as IDL SPEDAS and pySPEDAS (they
  were left out: the PADs were biased high, and the bins without counts
  NaN), and selects the channels with their lower edge at E -
  ``delta_energy_minus`` (it was E +). With the pitch angles of the files,
  the PADs equal pySPEDAS's.
- ``mms.eis_ang_ang`` bins the look directions on their own domains: the
  azimuth in [0, 360) and the polar angle in 6 bins from 10 to 170 deg,
  labelled at the bin centres. They were binned on [-180, 180] and [-80,
  80]: about 19 % of the samples were binned, at the polar angle + 90 deg.
  Each cell is the mean of its samples (the last sample was kept), NaN when
  empty (zeros).
- ``mms.eis_combine_proton_spec`` uses the channels of IDL SPEDAS 6.1: the
  lowest ExTOF channel is left out and every PHxTOF channel kept (it kept
  the lowest ExTOF channel and left out the top PHxTOF one). The energy
  widths are E minus the lower edge and the upper edge minus E (the edges
  were stored), and the samples are those at the times of both instruments
  (they could be paired across times).
- ``mms.eis_combine_proton_pad`` gives the PADs of the combined spectra,
  ``mms.eis_pad(mms.eis_combine_proton_spec(...))``, labelled with their
  energies (the columns were stitched from three channel selections and
  mislabelled, and the defaults raised on the 2016 tables); ``despin`` uses
  ``mms.eis_pad_spinavg`` (the PHxTOF spins were used on the ExTOF times).
- ``mms.eis_spec_combine_sc`` pairs the full-resolution spectra of the
  spacecraft at the same sector, nearest in time within half a spin (the
  sector gives the look direction), and the spin averages at the nearest
  time. They were paired by index, as IDL SPEDAS and pySPEDAS, which matches
  neither the times nor the look directions (61 to 86 deg apart on MMS2-4,
  2017-07-11). The sum is NaN where no spacecraft has data (it was 0).
- ``mms.eis_pad_combine_sc`` pairs the PADs at the nearest time and the
  closest energy channel, on the energies of the spacecraft with the fewest
  channels. They were paired by index, at the means of the energies of the
  same index, and different numbers of channels raised ``ValueError``. NaN
  without data (it was 0).
- ``mms.eis_skymap_combine_sc`` combines the skymaps at the nearest spin of
  each spacecraft (by index, one missing spin shifted the next ones), with
  the energy widths per channel; NaN where no spacecraft has data (the sum
  was 0). ``mms.eis_skymap`` gives the energy widths in eV, as the energies
  (they stayed in keV).
- ``mms.eis_moments`` sums over the energy channels and their widths: the
  integral by Simpson's rule from the first to the last channel centre left
  out the half channels at both ends (n was 0.39 to 0.94 of the partial
  density for 10 to 100 keV protons, 0.93 to 1.01 now; checked against
  ``mms.psd_moments`` on FPI-DIS, 0.996). With a background, the pressure is
  P - ``p_bg`` (it was reduced by 3 ``p_bg``).

Other fixes
^^^^^^^^^^^

- ``mms.eis_spec_combine_sc`` works on the outputs of ``mms.eis_omni``
  (``KeyError: 'energy_dplus'``) and with a sector without samples on a
  spacecraft (``IndexError``).
- ``mms.eis_skymap`` works with a single time (``IndexError``).
- The spin averages order the spins in time across the wrap of the spin
  counter (about every 15 days; ``numpy.unique`` reordered them).
- ``mms.dpf2psd``, ``mms.def2psd``, ``mms.psd2def``, ``mms.psd2dpf`` and the
  ``vdf_*`` conversions leave out ``LABLAXIS`` when the units change (a PSD
  kept the "Intensity" label of the flux).

Documentation
^^^^^^^^^^^^^

- ``mms.eis_pad``: ``vec`` is the magnetic field in GSE, the frame of the
  look directions; the pitch angles are those of the particles (180 deg
  minus the angle between the look direction and the field); the bins are
  means for every unit (a telescope is in up to three bins).
- ``mms.eis_omni``: ``method``, "sum" for counts and "mean" for everything
  else, as ``mms.vdf_omni``; the plain mean of the telescopes, as IDL SPEDAS
  and pySPEDAS (a NaN in one telescope gives NaN, zeros count).
- ``mms.eis_spec_combine_sc``: ``method``; IDL SPEDAS sums counts.
- ``mms.eis_combine_proton_pad``: the difference with IDL SPEDAS
  ``mms_eis_combine_proton_pad`` (which averages the cross-over channels).
- ``mms.eis_moments``: ``n_bg`` and ``p_bg`` must be the background of EIS
  in the same partial moments, not the FPI-DIS background; they will be
  deprecated, as the EIS files have no background product.

2.5.7 (2026-10-29)
------------------

This release lets ``pyrf.magnetosphere`` take the solar wind directly, as
irfu-matlab, with its magnetopause on the same angles, makes
``pyrf.lowpass`` stable for low cutoffs, and makes ``pyrf.gradient``
differentiate each segment between data gaps separately. ``mms.get_hpca_dist``
pairs each sample with its own azimuth and the anode elevations, and
``mms.get_data`` loads 146 more variables of the MMS CMAD. ``pyrf.wavelet``,
``pyrf.autocorr``, ``mms.rotate_tensor``, ``pyrf.cotrans``,
``pyrf.optimize_nbins_2d``, ``pyrf.ebsp`` and ``pyrf.iso86012datetime`` are
faster, with the same results. It fixes the handling of unusual inputs in
``mms.reduce``, ``pyrf.dist_append``, ``mms.get_dist``, ``mms.psd_moments``,
``pyrf.autocorr`` and ``pyrf.optimize_nbins_1d/2d``.

Changed behaviour
^^^^^^^^^^^^^^^^^

- ``pyrf.lowpass``: the sampling frequency is ``f_s``, as in
  ``pyrf.resample`` and ``pyrf.wavelet``, by default that of the input;
  ``fhz`` still works, with a ``FutureWarning`` (removed in 3.0). An input
  with NaN raises ``ValueError`` (the whole output was NaN), as do a cutoff
  outside (0, f_s / 2) and 12 samples or fewer.
- ``pyrf.magnetosphere`` raises ``ValueError`` for a non-positive or NaN
  dynamic pressure, e.g. without OMNI data in ``tint`` (the curves were
  NaN), and for a Mach number of 1 or less with the bow shock.
- ``mms.reduce`` raises ``ValueError`` (was ``AssertionError``) for an
  unknown ``dim`` or ``base``.
- ``mms.get_hpca_dist`` returns one distribution per half-spin of
  ``azimuth``, with their times (``time``), NaN for the half-spins not
  complete in the data or without azimuths (they were zeros, and a single
  zero azimuth raised ``AssertionError``); ``ValueError`` (was
  ``AssertionError``) for non-monotonic times or no complete half-spin. The
  anode elevations are read from the file of the data by default, as in
  ``mms.hpca_pad``.
- ``pyrf.iso86012datetime`` raises ``ValueError`` for NaT (it was a
  ``strptime`` format error).

New features
^^^^^^^^^^^^

- ``pyrf.magnetosphere``: ``d_p``, ``b_z`` and ``m_a`` give the solar wind
  directly, as irfu-matlab ``irf_magnetosphere(model, Dp, Bz)``; identical
  to it for both models. The model name is case-insensitive.
- ``mms.reduce``: ``verbose=False`` turns off the progress bar.
- ``pyrf.gradient``: ``time_step`` sets the data gaps (steps over 3
  ``time_step``), by default the smallest time step but one, as
  irfu-matlab ``irf_gradient``.
- ``mms.get_data``: 146 more variables of the MMS CMAD:

  - FPI: heat flux and its error, error flags, energy range of the density
    extrapolation, DES pitch angle distributions (low, mid and high energy),
    partial pressure (DBCS), DES fast parallel and perpendicular temperatures;
  - EDP: quality and bitmask of the E field (``qualitye``, ``bitmaske``) and
    of the spacecraft potential (``qualityv``, ``bitmaskv``: they differ),
    slow ``dce`` error and E parallel;
  - FGM: flag, position in GSE and GSM;
  - MEC: quaternions to 11 frames, MLT, MLAT, dipole L, Kp, Dst, Earth and
    Moon eclipse flags, position in SM;
  - HPCA: pressure tensors (DBCS, as irfu-matlab ``P*_dbcs_hpca_*_l2``),
    parallel and perpendicular temperatures, spin-averaged B (GSE, GSM);
  - DSP: E and B power spectral densities (fast, slow);
  - EDI: E and drift velocity (DSL, GSE, GSM; survey);
  - SCM: survey (``b_gse_scm_srvy_l2``) and high-speed burst
    (``bschb_gse_scm_brst_l2``).

  The new frames are ``geo``, ``sm``, ``gse2000``, ``smpa`` and ``ssl``.
- ``mms.get_hpca_dist``: ``elevation`` (the anode elevations), ``source``
  and ``data_path`` (to read them), as ``mms.hpca_pad``.
- ``pyrf.iso86012datetime`` accepts tuples, as ``pyrf.iso86012datetime64``.

Fixes that change results
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``pyrf.gradient`` differentiates each segment between data gaps
  separately, as irfu-matlab ``irf_gradient`` (it differentiated across
  the gaps): first-order differences at the samples next to a gap, NaN for
  a single sample between gaps; unchanged without gaps. This also changes
  dB/dt in ``pyrf.st_diff``. ``irf_gradient``'s own segments are off by one
  and still differentiate across the gaps next to them.
- ``pyrf.magnetosphere``: the Shue et al. (1998) magnetopause is at the
  angles 0, 0.1, ..., 3.1 rad, as irfu-matlab (it was ``linspace(0, pi,
  31)``: 28 points instead of 29 at the default solar wind). The bow shock
  is unchanged.
- ``pyrf.lowpass``: the elliptic filter is applied in second-order
  sections. The (b, a) form was unstable for cutoffs below about 3e-5 of
  the Nyquist frequency (e.g., 0.1 Hz at 8192 Hz: outputs of 40 for an input
  of 1 over 16 s), and imprecise near it (4 % at 1e-4). The padding is that
  of MATLAB's ``filtfilt`` (12 samples; scipy's 15 changed the first and
  last samples).
- ``mms.get_hpca_dist``: the data of a half-spin are its 16 samples from the
  first one (start azimuth 0). They were shifted by one sample, as in SPEDAS
  ``mms_get_hpca_dist``: each sample had the azimuth of the next one (11.4
  deg off), and the first sample was the last of the previous half-spin. The
  polar angles are those of the anodes (they were 90 deg minus the anode
  numbers), and ``dphi`` and ``dtheta`` have the dimensions of ``phi``.
- ``pyrf.optimize_nbins_1d`` and ``pyrf.optimize_nbins_2d`` include
  ``n_max`` in the numbers of bins tried, as documented (it was left out).
- ``pyrf.autocorr`` computes in double precision: float32 inputs differed by
  about 1e-7.

Other fixes
^^^^^^^^^^^

- ``mms.reduce``: ``lower_e_lim`` accepts an integer (it raised
  ``TypeError``); ``dim`` and ``base`` are case-insensitive (``"CART"``
  failed); a time step without any energy channel above the spacecraft
  potential and ``lower_e_lim`` is NaN (it raised ``IndexError``).
- ``pyrf.dist_append`` (used by ``mms.get_data`` for the distributions)
  joins the energy widths and the half widths of the accumulations only
  when both skymaps have them, and leaves them out otherwise: a missing key
  raised ``KeyError``, the ``None`` of files without energy widths
  ``AttributeError``, and the half widths of the first skymap alone were
  kept with the wrong length.
- ``mms.psd_moments`` no longer warns "invalid value in sqrt" for electrons
  with the spacecraft potential above the lowest channel (unused speeds),
  nor divides by zero where the density is zero (the bulk velocity,
  pressure, temperature and heat flux are NaN there, as before).
- ``pyrf.lowpass`` keeps the attributes of the input.
- ``mms.get_dist`` leaves out the energy widths of the files without them
  (they were None, on which ``mms.reduce`` raised ``TypeError``).
- ``pyrf.autocorr`` accepts integer inputs (``UFuncTypeError``).
- ``pyrf.optimize_nbins_1d`` and ``pyrf.optimize_nbins_2d`` leave out the
  non-finite samples (a NaN raised ``ValueError``), and raise a clear
  ``ValueError`` for constant inputs and x and y of different sizes.
- ``mms.tokenize`` no longer lists the parameters without data (``nbge``,
  ``pbge``, ``defbge``: no FPI DES background; ``tsi``, ``tse``: computed in
  irfu-matlab, the trace of ``ti_dbcs`` or ``te_dbcs`` over 3; ``epar``: the
  ``e_par`` keys); they are invalid parameters.
- ``mms.feeps_active_eyes`` returns the SITL eyes after 16 August 2017 in
  order.
- Faster, with the same results: ``pyrf.wavelet`` (about twice, and 1 to
  1.5 GB less memory for 3 x 200000 samples and 200 frequencies: the
  frequencies were tiled to the size of the transform),
  ``pyrf.autocorr`` (``scipy.signal.correlate``: 65 times on 200000 samples),
  ``mms.rotate_tensor`` (10 to 15 times: all times at once),
  ``pyrf.cotrans`` (the dipole direction once per call, for the
  transformations through GSM and SM), ``pyrf.optimize_nbins_2d`` (15 to 20
  times: about 30 s for 100 x 100 numbers of bins and 1e5 samples),
  ``pyrf.ebsp`` (one SVD per frequency: about 25 %) and
  ``pyrf.iso86012datetime`` (about 100 times).

Documentation
^^^^^^^^^^^^^

- ``pyrf.lowpass``: what it does (it said "low or highpass filter ... and
  subtract from the original").
- ``mms.reduce``: ``sc_pot``, ``lower_e_lim`` and ``verbose``.
- ``pyrf.ts_append``: it returns None for two None inputs (the start of
  accumulation loops).
- ``pyrf.cart2sph``: the azimuth is in (-pi, pi] (it said [0, 2 pi]), and
  the second angle is the polar angle from the z axis.
- ``mms.feeps_active_eyes``: SITL gives the top eyes 5, 11 and 12 for the
  ions too before 16 August 2017, as IDL SPEDAS and pySPEDAS.
- ``mms.get_hpca_dist``: the dimensions of the inputs and outputs.
- ``pyrf.autocorr``: the mean is not removed.

2.5.6 (2026-10-28)
------------------

This release adds single-spacecraft interferometry with the MMS electric
field probes: ``mms.probe_fields`` gives the fields between the six probes,
and ``mms.fk_power_spectrum`` the frequency-wave number power spectrum from
one probe pair. ``pyrf.c_4_v`` and ``pyrf.c_4_v_xcorr`` can give the
velocity in the frame of the positions, as irfu-matlab. ``pyrf.resample``
handles NaN with the cubic and quadratic interpolations, and ``pyrf.mva``
with all its methods. It fixes ``pyrf.integrate``, ``pyrf.mva``,
``pyrf.solid_angle`` and ``mms.fk_power_spectrum_4sc``, checked against
irfu-matlab and exact references.

Changed behaviour
^^^^^^^^^^^^^^^^^

- ``pyrf.resample``: the interpolation methods are the kinds of scipy's
  ``interp1d`` (``"linear"``, ``"cubic"``, ...), and other names raise
  ``ValueError`` (they raised ``NotImplementedError`` from scipy), with a
  hint for MATLAB's ``"spline"``, which is ``"cubic"`` here. With
  ``"cubic"`` or ``"quadratic"``, each component is interpolated through
  its finite samples, as MATLAB ``interp1`` (identical to 1e-16): a NaN
  gave NaN at all the times.
- ``pyrf.mva``, ``pyrf.integrate``, ``pyrf.solid_angle`` and
  ``pyrf.struct_func`` raise ``TypeError`` or ``ValueError`` for invalid
  inputs (were ``AssertionError``, or errors from NumPy or xarray).

New features
^^^^^^^^^^^^

- ``mms.probe_fields``: the fields from the potentials of the six probes of
  EDP (``v6_edp_brst_l2``), for probe interferometry: the single-probe
  fields (each probe against the spacecraft potential estimated by the
  other probes), the diagonal fields between neighbouring spin-plane probes
  and the double-probe fields, as irfu-matlab ``mms.fk_powerspectrum`` and
  ``mms.fk_powerspec_E80``. The channel delays of the burst sampling are
  corrected by a DFT shift (or by interpolation, or not), and the boom
  lengths can be the effective ones of Pedersen et al. (1998).
- ``mms.fk_power_spectrum``: the frequency-wave number power spectrum from
  the probes of one EDP pair (1-2, 3-4 or 5-6) aligned with B, from
  irfu-matlab ``mms.fk_powerspectrum`` (Graham et al. 2016), as a
  ``DataArray`` (f, k) for ``mms.estimate_phase_speed``. It agrees with
  irfu-matlab on MMS1 data of 2017-05-05 (wave numbers to 3e-5 or better).
  With ``field=False`` (the probe potentials instead of the single-probe
  fields), irfu-matlab's wave numbers are wrong (the second potential
  enters with a minus sign, which adds pi to the phase, over half the
  separation of the potentials); here the two potentials are used with
  their separation.
- ``pyrf.c_4_v`` and ``pyrf.c_4_v_xcorr``: ``frame="positions"`` gives the
  velocity in the frame of the positions (e.g., GSE), with each position at
  its own crossing time, as irfu-matlab ``c_4_v`` (checked on a drifting
  tetrahedron). The default, ``"spacecraft"``, is unchanged: the positions
  at the first time, the velocity relative to the spacecraft.

Fixes that change results
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``pyrf.mva`` with ``"<bn>=0"``: the third eigenvalue is the constrained
  minimum variance, the root of the constraint polynomial, as irfu-matlab
  ``irf_minvar`` (it was the variance along the normal of the field without
  its normal component, zero). The frames are unchanged.
- ``pyrf.integrate``: the times are in seconds from the first one, as
  irfu-matlab ``irf_integrate``. They were converted to float64 seconds
  since 1970, rounded to 256 ns: 6e-5 of a time step at 256 Hz, and 2e-5
  on ``int_e_dt`` and ``corr`` of ``pyrf.match_phibe_dir``.
- ``pyrf.solid_angle`` uses the formula of Van Oosterom and Strackee
  (1983) on the directions of the inputs. The spherical excess with the
  law of cosines (irfu-matlab ``irf.solidangle``) lost all precision for
  small triangles (a relative error of 0.2 for sides of 0.01 rad, and the
  wrong sign for 1e-5 rad), and gave NaN for vectors that are not unit
  vectors and for degenerate triangles.
- ``mms.fk_power_spectrum_4sc``: the wave numbers are the centres of the
  bins of the power maps. ``linspace(-k_max, k_max, num_k)``, as
  irfu-matlab, was up to half a bin off, and so the averages and standard
  deviations of k in the attributes (by 5-8 % on a test case with
  ``num_k=100``). The power maps are unchanged.

Other fixes
^^^^^^^^^^^

- ``pyrf.mva`` ignores NaN samples in the averages, as irfu-matlab (one
  raised ``LinAlgError``).
- ``pyrf.integrate`` accepts two samples, ``(time, 1)`` arrays and tensors
  (they raised), and keeps the attributes, with the units times s.
- ``pyrf.struct_func`` with a single scale (it raised ``ValueError``), also
  given as an integer.
- ``mms.lh_wave_analysis`` with B along x: the plane perpendicular to B is
  built from y instead of x (B x x vanished and all the outputs were NaN,
  as in irfu-matlab).
- ``pyrf.st_diff`` sets ``COORDINATE_SYSTEM`` to ``"lmn"`` on its output.

Documentation
^^^^^^^^^^^^^

- ``pyrf.st_diff``: the formula it computes, the velocity along each of l,
  m and n, valid in the frame of the minimum directional derivative, what
  ``lmn`` is and the units.
- ``pyrf.c_4_grad``: the layout of the gradient, ``"bdivb"`` and
  ``"curv"``, whose average unit vector is not normalised, as irfu-matlab
  (this partly compensates the error of the linear estimate).
- ``pyrf.sph2cart``: the second angle is the colatitude, as returned by
  ``pyrf.cart2sph`` (MATLAB's ``sph2cart`` takes the elevation).
- ``pyrf.dec_par_perp``: ``alpha`` is in radians (degrees in irfu-matlab),
  and zero without ``flag_spin_plane``.
- ``pyrf.struct_func``: the absolute increments, without the nonexistent
  ``ncut``.

2.5.5 (2026-10-23)
------------------

This release completes ``pyrf.c_4_v_xcorr`` for noisy signals, with a
low-pass inside the fit (``f_max``) and errors in speed and direction that
include the correlations of the delays, and adds an example of
four-spacecraft timing by cross-correlation. It reads the MMS DEFQ files,
and fixes ``mms.fft_bandpass``, which could halve a component at a cutoff.

Changed behaviour
^^^^^^^^^^^^^^^^^

- ``pyrf.c_4_v_xcorr``: ``d_speed``, ``d_theta`` and ``e_theta`` follow
  from the slowness covariance ``dm2`` (Vogt et al. 2011, eqs. 19-20),
  with the correlations of the delays, and differ from those of
  irfu-matlab either way. They followed irfu-matlab, with the formulas of
  the simplified case of uncorrelated and identical errors (eqs. 32-33) and
  the scalar time error ``trace(sqrt(dtau2)) / 4``, which ignores the
  correlations, never zero with a pattern made of the four signals. On
  synthetic structures, the delays are anti-correlated (about -1/3), and
  these errors were 15-30 % smaller than the spread of the speed and the
  direction over noise draws (now 0-17 %). On the interplanetary shock of
  2018-01-08, the delays are strongly correlated: the speed error goes from
  34.3 to 15.4 km/s (irfu-matlab: 34.1), and the direction errors from 3.4
  and 3.9 deg to 2.2 and 4.8 deg. Both agree for uncorrelated and identical
  errors.
- ``pyrf.dec_par_perp`` checks the components of its inputs (3 for
  ``b_bgd``, 3 for ``inp``, or 2 or 3 with ``flag_spin_plane``, of which x
  and y are used) and raises ``TypeError`` or ``ValueError`` (was
  ``AssertionError``; an ``inp`` of 2 components failed in a broadcast).

New features
^^^^^^^^^^^^

- ``pyrf.c_4_v_xcorr``: ``f_max`` low-passes the four signals and the
  pattern inside the fit, with the same zero-phase filter for all (a
  raised-cosine roll-off from ``f_max`` / 2 to ``f_max``): the delays are
  unchanged. At or a little above the top of the band of the structure, it
  brings the spread of the delays over noise draws within 1.0-1.4 of the
  reported errors (1.7-2.3 without), and the false matches of
  ``fit="none"`` on a noisy electron hole from 16 % to 1-2 % of the draws.
- ``mms.load_ancillary``, ``mms.list_files_ancillary``,
  ``mms.list_files_ancillary_sdc``, ``mms.download_ancillary`` and
  ``mms.copy_files_ancillary`` find the DEFQ (and PREDQ) files, one for the
  four spacecraft (``MMS_DEFQ_*``, ``ancillary/mms/defq``; they were looked
  for as ``MMS<n>_DEFQ`` and never found): ``load_ancillary("defq", ...)``
  gives the tetrahedron quality factor ``q``, ``scale`` and
  ``maneuver_flag`` every 30 s, for any ``mms_id``.
- ``mms.fft_bandpass`` accepts integer frequencies (they raised
  ``TypeError``).

Other fixes
^^^^^^^^^^^

- ``mms.fft_bandpass``: the frequencies of the FFT bins are those of
  irfu-matlab ``mms.fftbandpass`` (they were spaced by 2 f_nyq / (N - 1)):
  the positive and negative frequencies of a component within a bin of a
  cutoff could fall on either side of it, which halved the component.
- ``mms.load_ancillary`` reads the files without data (``No data exists``,
  e.g. DEFQ of 2017-10-02) as no rows (it raised ``ValueError``).
- ``mms.hpca_spin_sum`` raises a clear ``ValueError`` with fewer than two
  spin starts (one raised ``IndexError``).
- ``pyrf.datetime642unix`` accepts an empty input (it raised
  ``IndexError``).
- ``pyrf.mva_gui``: the Reset button works (every click raised
  ``TypeError``).
- ``pyrf.ts_spectr`` with energies varying with time keeps ``attrs`` on the
  data of the Dataset, with ``TENSOR_ORDER`` (they were dropped).

Documentation
^^^^^^^^^^^^^

- A new example notebook, four-spacecraft timing by cross-correlation: the
  electron hole of Fig. 2 of Kamaletdinov et al. (2021), on 2017-08-04,
  with the delays, the velocity and its errors from ``pyrf.c_4_v_xcorr``,
  and a first guess of the delays in a train of electron holes.
- ``pyrf.c_4_v_xcorr``: the use of ``f_max`` against the noise, and the
  equations of Vogt et al. (2011) behind each error.
- ``pyrf.edb``: ``angle_lim`` is described the right way (with ``"E.B=0"``,
  Ez is computed where B is more than ``angle_lim`` above the spin plane),
  with the formula of each method.
- ``mms.hpca_spin_sum``: the samples after the last spin start are not used,
  as pyspedas.

2.5.4 (2026-10-22)
------------------

This release adds ``pyrf.c_4_v_xcorr``, four-spacecraft timing by
cross-correlation with error estimates, reads the MMS ancillary files from
the SDC, and lets ``psd_moments`` start the ion moments at FPI's
recommended energy channel. The look of the example notebooks becomes the
pyrfu style (``plot.use_pyrfu_style()``, ``plt.style.use("pyrfu")``). All
the example notebooks read their data from AWS, without ``db_init``, in
that style, and the results of several of them are fixed.

Changed behaviour
^^^^^^^^^^^^^^^^^

- ``plot.use_pyrfu_style()`` applies the ``pyrfu`` style sheet on
  matplotlib's defaults: the look of the example notebooks (pyrfu colour
  cycle, ``Spectral_r`` colormap, minor ticks, mathtext in DejaVu Sans,
  frameless legends in the upper right corner). Its colormap, and that of
  ``plot.set_color_cycle("pyrfu")``, is ``Spectral_r`` (was
  ``hot_white_desaturated``), and so the default of ``plot_spectr``. The
  ``pyrfu.mplstyle`` sheet (``plt.style.use("pyrfu")`` once ``pyrfu.plot``
  is imported) is this style: the former serif sheet (labels of size 8,
  ticks in, eps at 600 dpi) is removed.
- ``mms.load_ancillary`` reads the files from the resource of the
  configuration (``source="default"``, see ``mms.db_init``), or from
  ``source``: with ``"sdc"`` or ``"aws"`` (AWS has no ancillary files) it
  reads them from the SDC, where it read only local files. Use
  ``source="local"`` for the former behaviour.

New features
^^^^^^^^^^^^

- ``pyrf.c_4_v_xcorr``: four-spacecraft timing by cross-correlation, from
  irfu-matlab ``c_4_v_xcorr`` and ``irf_4_v_xcorr_errors``, for boundaries,
  current sheets, shocks or localised structures. The delays to a common
  pattern are continuous (band-limited shift of the signals), and the
  velocity and its errors follow Vogt et al. (2011). Scalar or vector
  signals, with cadences that may differ slightly. Without noise, the delays
  are within 5e-5 sample; on the interplanetary shock of 2018-01-08,
  356.8 +- 34.3 km/s (irfu-matlab: 357.5 +- 34.1). ``fit`` chooses what is
  fitted per spacecraft besides the delay: nothing (``"none"``, the default,
  as irfu-matlab), a scale (``"scale"``, for structures seen with amplitudes
  differing by more than a factor of 2, e.g. an electron hole crossed far
  from its centre: unscaled, the delay is wrong, with a poor correlation),
  or a scale and an offset (``"scale_offset"``, not for boundaries). Unlike
  irfu-matlab, no start guess is needed and the delays are not pulled
  towards half-samples (by up to 0.11 sample with linear interpolation).
  With noisy signals, the errors are underestimated unless the signals are
  low-passed to the band of the structure (see its Notes).
- ``mms.load_ancillary`` reads the ancillary files from the SDC (DEFATT,
  DEFEPH; not yet DEFQ, whose files cover the four spacecraft): the latest
  version of each file covering the interval.
- ``mms.psd_moments``: ``en_channels`` accepts a time series of first
  channels, e.g. FPI's recommended energy index ``partidxi_fpi_brst_l2``:
  at each time, the channels from it to the last. FPI's ion moments start
  at this index (0 in dense plasma, up to 11, about 100 eV, in the tenuous
  plasma sheet), and from it ``psd_moments`` agrees with FPI's ion density
  within 4 % on the intervals tested (with all the channels: 3.8 times on
  2017-07-11, MMS1). ``remove_idist_background`` is another correction.
- ``mms.get_data`` keys:

  - ``vpsp_edp_{brst,fast,slow}_l2``: the probe-to-spacecraft potential
    averaged over the probes, as irfu-matlab ``Vpsp_edp_*_l2``;
  - ``partidx{i,e}_fpi_{brst,fast}_l2``: the recommended energy index of
    the FPI partial moments.

- ``plot.plot_spectr`` accepts ``clim`` as a tuple or an array (only a list
  was used; others were ignored silently).
- ``plot.plot_heatmap`` and ``plot.annotate_heatmap`` accept lists (they
  passed the type check, then failed).

Other fixes
^^^^^^^^^^^

- ``plot.plot_projection``, ``plot.ion_brazil_plot_thresh``,
  ``plot.plot_clines``, ``plot.plot_reduced_2d`` and ``plot.plot_surf``
  draw their colour bar or legend in the figure of their axis (they used
  the current figure, e.g. after ``plt.figure()``).
- ``plot.plot_spectr`` with ``pad=0``: the colour bar is against the axis
  (0 was taken as 0.01).
- ``plot.plot_line`` and ``plot.plot_spectr`` use the concise date format
  for times in any unit (only nanoseconds; xarray keeps other units).
- ``plot.pl_tx`` draws the grid as ``plot_line`` (darker panels).
- Reloading ``pyrfu.plot`` (e.g. autoreload in notebooks) raised
  ``ValueError``: the colormaps are registered once.
- ``mms.eis_pad`` and ``mms.eis_pad_spinavg`` no longer warn for the empty
  pitch-angle bins (``Mean of empty slice``).
- The MMS configuration is read again when ``MMS_CFG_PATH`` points to
  another file with the same modification time (the cache was keyed by
  the time only).

Documentation
^^^^^^^^^^^^^

- The example notebooks read their data from AWS, without ``db_init``
  (``load_ancillary`` from the SDC), all in the pyrfu style, with panel
  labels. Fixed results:

  - ``ohmslaw``: the sign of the convection fields (E + V_i x B now matches
    the right-hand side);
  - ``electron_psd``: the units of the phase-space density and the 90 deg
    bin;
  - ``b_e_j``: \|div B\|/\|curl B\| (computed after scaling J), and a note
    on the quality of the curlometer estimate;
  - ``edr_signatures``: the spike filter (the first sample was compared
    with the last) and the units of E'.J;
  - ``particle_distributions``: the 90 deg bin and the angle labels;
  - ``reduced_electron_dist``: reduced in the (B, E x B) frame it is
    labelled with, with the spacecraft potential;
  - ``reduced_ion_dist_para``: the fitted time selected by value;
  - ``particle_pad``, ``walen_test``, ``hpca``: the densities plotted and
    labelled, and the HPCA flux units;
  - ``particle_deflux``: the potential lines below the energy range
    removed.

- ``plot.colorbar`` returns the colour-bar axis, ``plot.plot_spectr``
  ``(axis, cax)`` (not ``fig, axs, caxs``), and ``plot.plot_projection``
  takes velocities in km/s, with a colour bar on the right and matplotlib's
  colormap by default; ``plot.ion_brazil_plot_thresh`` documents
  ``legend`` and ``colors``.
- ``mms.psd_moments``: how its ion moments compare with FPI's.

Development
^^^^^^^^^^^

- ``codecov.yml``: 100 % coverage of the patch, and no drop of the project
  coverage (merges are not blocked).
- ``test_plot.py`` no longer opens 20 figures at import, and every test
  closes its figures.

2.5.3 (2026-10-21)
------------------

This release is the first published to PyPI by trusted publishing. It
fixes ``ebsp`` with ``de_dot_b0``, the time derivatives (``gradient``,
``sliding_derivative``), the reading of MMS files around midnight and of
overlapping ancillary files, and the joining of time series from several
files. It accepts the irfu-matlab variable names in ``get_data``, and
distributions with ``phi`` as ``(phi,)`` in more routines. Several fixes
change results: see `Fixes that change results <v2.5.3 fixes_>`_.

Changed behaviour
^^^^^^^^^^^^^^^^^

- ``mms.tokenize`` (and so ``mms.get_data``) accepts variable names in any
  case, as the irfu-matlab names (``"B_gse_fgm_srvy_l2"``); a name without
  4 or 5 parts and an unknown parameter raise ``ValueError``, as documented,
  instead of ``AssertionError``.
- ``pyrf.read_cdf`` raises ``FileNotFoundError`` for a missing or unreadable
  file (it raised ``AttributeError``).
- ``pyrf.sliding_derivative`` raises ``ValueError`` for an invalid
  ``t_units`` or ``method``, and ``pyrf.ts_convolve`` ``TypeError`` for an
  input that is not a ``DataArray`` (both raised ``AssertionError``).
- ``pyrf.ts_convolve`` returns float64 for integer data (see below).

New features
^^^^^^^^^^^^

- ``pyrf.datetime2iso8601`` accepts ``numpy.datetime64`` times (scalars and
  arrays, e.g. the times of a time series).
- ``pyrf.iso86012datetime64`` accepts a single str and tuples, and times
  ending with ``"Z"`` without warning.
- ``pyrf.ts_time`` accepts any float type and ``datetime64`` in any unit.
- ``pyrf.avg_4sc``, ``pyrf.c_4_grad`` and ``pyrf.c_4_j`` accept a tuple of
  the four spacecraft, as their type hints say.
- ``pyrf.sliding_derivative`` accepts vectors and tensors.
- ``pyrf.ebsp`` accepts ``fac_matrix`` as an array, ``(n, 3, 3)`` on the
  times of ``delta_b`` or a constant ``(3, 3)``, as documented (only a time
  series of matrices worked).
- ``mms.make_model_vdf``, ``mms.calculate_epsilon``, ``plot.plot_ang_ang``
  and ``pyrf.ts_skymap`` accept distributions with ``phi`` as ``(phi,)``,
  as ``vdf_omni`` and ``get_pitch_angle_dist`` (``ts_skymap`` repeats it at
  all times).

.. _v2.5.3 fixes:

Fixes that change results
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``pyrf.ebsp`` with ``de_dot_b0`` and FAC (the default): E was rotated to
  field-aligned coordinates with B0 resampled to wrong times (in 1970), so
  the E powers, Poynting flux and polarization were wrong whenever B0
  varies (60-80 % perpendicular, x100 parallel, for B0 rotating by 90 deg).
  Identical to irfu-matlab's ``irf_ebsp`` now.
- ``pyrf.gradient`` divided by the median time step: wrong next to data
  gaps (65 times the slope around a 1 s gap in 128 Hz data) and with
  jittered times (15 % for +-1 ms at 128 Hz). It uses the actual time
  steps; unchanged for even sampling. ``pyrf.st_diff`` uses it for dB/dt.
- ``pyrf.sliding_derivative``: times as float64 epochs were rounded to
  256 ns, an error of 0.1 % on the derivative of 8192 Hz data.
- ``pyrf.ts_convolve`` truncated the convolution of integer data (a moving
  average of counts of 0 and 1 gave zeros).
- ``pyrf.ts_append`` (used to join files in ``mms.get_data``,
  ``mms.db_get_ts``, ``mms.get_eis_allt`` and the Solar Orbiter readers):
  a time-dependent 2-D attribute was appended along its second dimension
  (or raised ``ValueError``), a 0-d array became two values and a constant
  table was doubled. As ``time_clip``, only the arrays with the times along
  the first dimension are appended.
- ``mms.list_files`` (local data) missed the file that starts the day
  before the time interval, e.g. a burst segment across midnight: no file,
  or the first minutes of data missing. As ``list_files_aws``, the search
  starts the day before.
- ``mms.load_ancillary`` returned repeated times: DEFATT files repeat some
  times by design and consecutive files overlap (by about 5 minutes). As
  irfu-matlab, the last repeated row of a file is kept and, in an overlap,
  the earlier file (identical to irfu-matlab on MMS1 5-7 Jul 2017).
- ``pyrf.ts_time`` truncated Unix times to the nanosecond (6.5e-5 s gave
  64 999 ns), and ``pyrf.extend_tint`` the extension (0.0157 s added
  15 699 999 ns); both are rounded.
- ``pyrf.pvi_4sc`` gave NaN shear angles for parallel fields (cosines just
  above 1 from rounding) instead of 0.

Other fixes
^^^^^^^^^^^

- ``pyrf.sliding_derivative`` raised ``AttributeError`` with
  ``t_units="s"``, and a method in another case (``"Window"``) gave only
  NaN.
- ``pyrf.ts_skymap`` raised ``IndexError`` for a single time without
  ``energy1``; an invalid ``energy1`` named ``energy0`` in the error.
- ``pyrf.extend_tint`` returns ``str`` items, as documented (they were
  ``numpy.str_``).
- ``pyrf.calc_fs`` and ``pyrf.calc_dt`` give NaN without warnings for a
  series of one sample (documented; ``resample`` and ``avg_4sc`` accept such
  series).
- ``pyrf.magnetosphere``: the bow shock downloaded the OMNI data twice.
- ``mms.list_files`` walks each directory once (the month directory of
  survey data was walked for every day of the interval).
- The SDC file listings no longer request a doubled slash
  (``.../api/v1//file_info/``), and ``mms.list_files_sdc`` imports
  ``urllib.parse`` itself.

Documentation
^^^^^^^^^^^^^

- ``pyrf.plasma_calc``: the keys of the output (``w_pe``, ``v_a``, ``l_i``,
  ... none of the documented ones existed), and the example takes a third
  of the trace as the scalar temperature (it used the trace).
- ``pyrf.pres_anis``: the example passes the tensor in field-aligned
  coordinates (it passed the GSE one); the units are given.
- ``pyrf.wave_fft``: the return order (frequencies, time, spectrogram) and
  the required ``window``.
- ``pyrf.magnetosphere``: the example runs (it was irfu-matlab's, a
  ``TypeError``); both models and the default solar wind are documented.
- ``pyrf.read_cdf``: what it returns (epochs not decoded, size-1 dimensions
  removed).
- ``pyrf.pvi_4sc``: Delta B_ij = B_i - B_j (was B_i - B_i), the outputs
  are lists of the 6 pairs, the angles in radians.
- ``pyrf.sliding_derivative`` documents ``method``, ``pyrf.ts_convolve``
  ``mode``, and ``pyrf.ebsp`` the forms and rows of ``fac_matrix``; the
  ``pyrf.extend_tint`` example runs.

Development
^^^^^^^^^^^

- Releases are published by trusted publishing: the workflow checks that
  the tag is the package version, builds and tests the wheel, and uploads
  without an API token. The TestPyPI upload on every push is removed.
- The PyHC checks (PHEP 3 compliance and PyHC environment compatibility)
  run on ``master`` and ``devel``, from PR #38 by S. Polson.
- A CI job tests the lowest dependency versions that ``pyproject.toml``
  allows, on Python 3.12.
- The coverage is measured on the library only (93 %). The SDC download,
  the local copies and ``db_get_ts`` are tested offline (mocked SDC,
  temporary configuration), and so are the plot functions that had no
  tests; the colour cycle test never ran.

2.5.2 (2026-10-20)
------------------

This release fixes the routines that returned wrong values in common use
(``dft_time_shift``, ``make_model_kappa``, ``scpot2ne``, the ``shock_normal``
Monte Carlo, ``wavelet`` with data gaps, ...), ports the Solar Orbiter TNR
reader from irfu-matlab, and adds multi-spacecraft and probe timing
utilities (``mms4_displacement``, the ``V6`` probe potentials). Each fix is
checked against irfu-matlab, an analytic case or real data. Several fixes
change numerical results: see `Changed behaviour <v2.5.2 changed_>`_ and
`Fixes that change results <v2.5.2 fixes_>`_ before updating an analysis.

.. _v2.5.2 changed:

Changed behaviour
^^^^^^^^^^^^^^^^^

These changes can require updating your code.

- ``pyrf.iso86012unix`` returns Unix seconds (float64, always 1-D, NaN for
  NaT), as its name says and as before 2.3.7; it returned ``datetime64[ns]``,
  the output of ``iso86012datetime64``.
- ``pyrf.rotate_tensor``: the perpendicular rotation (``perp="pp"`` or
  ``"qq"``) is for field-aligned coordinates only, as irfu-matlab. A tensor
  rotated to user-given axes (``"rot"``, e.g. LMN) was also rotated about the
  first axis; an explicit ``"pp"`` or ``"qq"`` with another frame is ignored
  with a warning.
- ``mms.fk_power_spectrum_4sc``: ``k_y_f`` and ``k_z_f`` are on the
  dimensions ``k_y`` and ``k_z`` (both were on ``k_x``).
- ``pyrf.medfilt`` returns float64 (it returned float32).
- ``pyrf.mean_bins`` and ``pyrf.median_bins`` use the bins of
  ``numpy.histogram``: the samples on the bin edges, including ``min(x)`` and
  ``max(x)``, are counted (the first and last bins were often NaN), and the
  non-finite pairs are left out.
- ``pyrf.int_sph_dist`` (and ``mms.reduce``): ``weight="log"`` gives the bins
  1 + ln(f / min f) particles, whatever the units of the distribution. It was
  equal to ``"lin"`` for phase-space densities in SI units.
- ``solo.read_tnr`` returns every sweep, without the subtraction of the
  integral over frequency, and ``solo.read_tnr`` and
  ``solo.read_lfr_density`` read only the latest version of each daily file
  (see `Fixes that change results <v2.5.2 fixes_>`_).

Deprecations
^^^^^^^^^^^^

- ``pyrf.filt``: the default ``f_max`` = 1 Hz adds a 1 Hz low-pass to
  ``filt(x, f_min)``. It warns (``FutureWarning``) and will be ``f_max = 0``
  (high-pass only) in 3.0, as irfu-matlab ``irf_filt``; give ``f_max`` to
  keep the current behaviour without the warning.

New features
^^^^^^^^^^^^

- ``mms.mms4_displacement``: the separations of the six pairs of spacecraft
  along and across the field, and their angle to B (irfu-matlab
  ``mms.mms4_displacement``, to 1e-7).
- ``mms.get_data``: the keys ``v6_edp_{brst,fast,slow}_l2``, the six
  probe-to-spacecraft potentials (irfu-matlab ``V6_edp_*_l2``, identical
  samples).
- ``pyrf.pid_4sc_err`` (the error on Pi-D, Roberts et al., 2023) is exported
  by ``pyrfu.pyrf`` and documented.
- ``mms.dft_time_shift``: new ``f_s`` argument (irfu-matlab ``Freq``).
- ``mms.make_model_kappa``: new ``sc_pot`` argument, as ``make_model_vdf``.
- ``pyrf.shock_normal``: new ``random_state`` in the Monte-Carlo ``spec``.

.. _v2.5.2 fixes:

Fixes that change results
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``mms.dft_time_shift`` shifted by whole samples and the wrong way, with a
  complex output: as irfu-matlab, it now shifts by any fraction of a sample
  (2e-15 from irfu-matlab).
- ``mms.make_model_kappa``: the units (1e6 too small), the particle
  directions and the azimuths of each time are fixed; the model gives back
  n, V and T through ``psd_moments`` and agrees with irfu-matlab
  ``generate_dist(..., 'kappa')`` to 2e-9.
- ``mms.scpot2ne`` fitted the second photoelectron population with the wrong
  sign (61 % from irfu-matlab on one hour of MMS1 data; now 4e-8).
- ``pyrf.shock_normal``: the time-series (Monte Carlo) branch ignored its
  draws; each of the ``n`` draws (default 100, as documented) now goes
  through the single-value calculation, as irfu-matlab.
- ``pyrf.c_4_v``: the crossing delays of ``datetime64`` inputs are taken in
  integer ns; float seconds since 1970 resolve only ~0.24 µs, 2–3 km/s on
  the velocity of structures crossing the tetrahedron in less than 1 ms.
- ``pyrf.wavelet``: one NaN (or inf) made the whole transform NaN. As
  irfu-matlab, the gaps are set to 0 for the transform and NaN at their own
  times only; ``mms.fk_power_spectrum_4sc`` now works on data with gaps.
- ``mms.estimate_phase_speed`` weighted the fit by the inverse of its
  weights (the weakest points the most).
- ``pyrf.pid_4sc`` mixed the time bases of P and V (``ValueError``, or
  misaligned on shifted grids): P is resampled onto the time base of V.
  ``pyrf.pid_4sc_err`` was NaN wherever Pi or D is 0 (every sample for an
  isotropic pressure).
- ``pyrf.medfilt`` rounded to float32 (0.031 off for values of 1e6).
- ``solo.read_tnr``, port of irfu-matlab ``read_TNR.m``: the first and last
  sweeps were dropped, the sweeps were split at every decrease of the sweep
  number (uint32), the times got out of step with the spectra, and the
  subtraction of the integral over frequency gave negative powers and
  all-NaN spectra (6.5 % on 2022-03-01). A file without the sensor is
  skipped.
- ``solo.read_tnr`` and ``solo.read_lfr_density`` read every version of a
  daily file, and the ``-cdag`` and public files, so a day could be read two
  or three times; ``read_lfr_density`` also read the 10-second product. They
  read the highest version of each date (the public file on a tie), and
  ``read_lfr_density`` sorts its output in time.

Other fixes
^^^^^^^^^^^

- ``pyrf.histogram`` raised ``ValueError`` on any NaN or inf with the
  automatic range; the non-finite samples are left out.
- ``pyrf.date_str`` rejected the standard ``tint`` format
  (``"2019-09-14T07:54:00.000"``), and ``pyrf.iso86012timevec`` any time
  without 9 fractional digits.
- ``pyrf.medfilt`` failed on (time, 9) and (time, 1) series and on a float
  kernel size.
- ``pyrf.ebsp`` with ``de_dot_b0=True`` failed when the full B had another
  cadence than E and dB; it is resampled onto their time line.
- ``mms.fk_power_spectrum_4sc`` failed on vector fields (the first component
  is used, as documented) and on data gaps; ``k_perp`` could be NaN from
  rounding.
- ``mms.scpot2ne`` accepts a scalar temperature, as documented.
- ``pyrf.rotate_tensor`` warned that every ``"rot"`` matrix was left-handed.

Documentation
^^^^^^^^^^^^^

- The example notebooks no longer emit ``SyntaxWarning`` (invalid escape
  sequences in the LaTeX labels).
- ``CONTRIBUTING.rst`` and the installation page give Python 3.12–3.15 and
  an https clone URL.
- The units of ``mms.fk_power_spectrum_4sc`` (rad/m for positions in km) and
  the float64 resolution of ``pyrf.iso86012unix`` are documented.

Development
^^^^^^^^^^^

- CI runs on pull requests (and on pushes to ``master`` and ``devel``), with
  the test extra of the package, and a new job runs the tests of the built
  wheel installed in a clean environment.
- The tests are reproducible (seeded), with a 300 s timeout per test, a
  ``network`` marker, ``--strict-markers`` and ``--strict-config``, and
  targeted warning filters instead of ignoring every ``RuntimeWarning``.
- The docs build with current Sphinx (``docs/requirements.txt``: Sphinx
  9.1, pydata-sphinx-theme 0.22).
- Two Jupyter checkpoint files are no longer tracked.

Requirements
^^^^^^^^^^^^

- Building needs setuptools 77 or later; the license is declared as an SPDX
  expression (PEP 639), and the ``wheel`` pin is removed.
- New test dependency: ``pytest-timeout``.

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
