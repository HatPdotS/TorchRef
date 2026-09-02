Changelog
=========


Unreleased
----------
- Fixed the rotation function contracting **unconjugated** calc coefficients on MPS. ``torch.conj`` returns a lazy view carrying a conjugate *bit*, and MPS's batched complex matmul -- what the radial ``einsum`` lowers to -- ignores it, so the contraction was silently wrong: on 1DAW by 173% of ``|xi|`` max, which reordered the entire peak list and pushed the true orientation from rank 0 out of the top 200 while the top score moved only 0.1%. Materialised with ``resolve_conj()`` at the two batched-matmul sites. Elementwise ops, ``where``, ``index_add``, ``fft`` and 2-D matmul all honour the bit and only the batched path does not, which is too narrow to guard structurally, so the guard is the contraction's own value against a host-double reference
- The alignment package runs end to end on an accelerator without float64: 123 alignment tests including the slow rotation searches pass on MPS, where every one of them failed before. Five things stood in the way, four of them the same mistake -- casting and moving in two steps. ``.to(device).to(dtype)`` puts the caller's width on the device first, which throws for a double input on a backend that has none (14 sites, now one fused ``.to`` each); ``detect_zsymm`` widened the symmetry operators to double while they sat on the device; the expansion's host-side clustering keys left a host index to meet device values; and MPS's ``linalg.inv`` trips an internal contiguity assert on a transposed 3x3 view. The fifth is a kernel gap -- MPS implements no complex cumulative op -- so the azimuthal phase ladder is built by doubling instead of by ``cumprod``, measured at 5.7e-6 against ``cumprod``'s 4.1e-6 in complex64 over 5000 angles at L=101
- The anisotropy fit runs at the configured float dtype on the data's own device, instead of double on the host. Measured over the 16 datasets in ``tests/files/mtz``, float32 reproduces the double fit to 3.3e-5 relative in ``U`` and 4.5e-6 in the correction factor ``exp(+pi^2 s.U.s)`` it exists to produce -- the design matrix is a constant column beside ``2 pi^2 s.s`` terms of order 0.1-1, so there is no precision cliff for seven parameters to fall off. Unrotated searches on 1DAW, 3K7M, 2DQ6 and 4BX9 return the same peaks in the same order as before, with scores moved 1.8e-7 to 2.5e-4 relative. ``fit_anisotropy`` takes an optional ``device``; ``get_axis_order`` and ``hkl_symops_to_cartesian`` follow their inputs' width and place rather than forcing double
- The Wilson fit's IRLS solve is a Cholesky reuse plus two triangular solves, not ``lu_factor``/``lu_solve``. ``A`` is ``X^T W X`` plus a ridge, positive definite by construction, and MPS implements neither ``lu_solve`` nor ``cholesky_solve``, so the old path left the per-iteration solve to a CPU round trip: 1015 us against 114 us, on a 200k x 6 problem whose unavoidable ``XtW @ z`` is 966 us. Over the 16-dataset panel the two agree to 1e-5 relative in float32 and 1e-14 in float64 with identical iteration counts on every case. ``cholesky_ex`` reports rather than raises, and a non-positive-definite report falls back to a general solve, since a fully collinear basis is a thing this fit sees
- Dropped the two alignment tests that imported ``alignment_lab``: an external experiment is not the main package's to test, and they could not pass in a clean checkout. ``resolve_device`` is covered directly in ``tests/unit/utils/test_device_resolution.py``. Two test-side assertions that could not hold at the configured width were repaired -- ``<E^2> = 1`` now widens after the readback rather than on the device, and the weight's mean-one bound tracks the dtype's epsilon instead of a fixed 1e-9 that is below one float32 ulp
- Nothing in the alignment package puts float64 or complex128 on the compute device any more, so it runs on backends without double (MPS). The rotation function's radial sum, Wigner contraction and FFT accumulate at the configured complex dtype rather than one step wider; the Bessel ladder runs in its argument's dtype, kept in range by its rescaling; the expansion's exact clustering keys are formed on the host in double, which the device never sees; rotation-function inputs, the dense P1 transform and the shell sums use the configured float dtype. 30/30 poses at both windows, timings unchanged; the double-precision left is host-side 3x3 rotation algebra, the Wigner-d eigendecomposition and the anisotropy fit
- Every hard-coded dtype in the alignment package either moved to the configured dtype or carries a ``# dtype-ok:`` justification, as the dtype-conformance guard requires: three casts to double dropped from the empirical sigma_A ratio and the dense P1 transform, the translation set's resolution mask and peak translations use the configured float dtype, and the rest (index tensors, host-side 3x3 rotation algebra, the rotation function's double accumulation, the anisotropy fit) are annotated
- Removed ``DirectModelEvaluator``; the translation search evaluates an ordinary P1 ``ModelFT`` directly. With the model's grid derived lazily from cell, space group and ``max_res`` there was nothing left for the wrapper to do
- The molecular-replacement pipeline carries 10 rotation candidates by default instead of 25. With symmetry mates suppressed the rotation function's first peak is the true orientation in 50 of 50 pose-gated cells, and the panel is 30/30 at either depth. Warm on one EPYC 9335 node: 1DAW 0.42 s, 2DQ6 0.67 s, 6G9X 0.67 s, 3K7M 1.02 s, 4BX9 1.11 s per alignment
- The fast translation function accumulates only the upper triangle of symmetry pairs; the lower triangle is its conjugate mirror and the diagonal a constant. Half the scatter, which is the stage's cost on high-symmetry cells: 3K7M's translation stage 1.04 s to 0.72 s. ``MRSolution.candidate_index`` records each solution's position in the rotation function's list
- The placement loop re-orients one P1 copy of the search model in place per candidate and builds the placed model for the winner only, instead of copying the model three times per candidate. ``MRSolution.model`` is ``None`` for the other candidates; ``MolecularReplacementPipeline.place`` builds it on request. Warm on one EPYC 9335 node: 1DAW 0.85 s, 2DQ6 1.35 s, 6G9X 1.30 s, 3K7M 2.35 s, 4BX9 2.55 s per alignment (from 1.1, 1.9, 2.2, 2.7, 3.8)
- The three candidate-ranking scores (likelihood, analytical R, fast translation score) pick the same candidate in all 60 cells of a six-structure x ten-seed sweep measured on poses; every earlier figure quoted for them was rotation-only. The likelihood stays the default
- The rotation function suppresses symmetry mates when it picks peaks, so its shortlist is one entry per orientation. The point group composes on the right of a peak's rotation, ``R R_g`` -- measured on real peak lists, where the left orbit finds no coincident pairs and the right finds every mate (187 of the 300 pairs among 3K7M's top 25 were mates of each other). The lab's orbit-based truth rank defaulted to the left side, which is why it disagreed with coordinate superposition. Placements unchanged, 30/30 at both windows
- Fixed ``empirical_sigma_a`` taking the level of the observed-to-calculated Wilson-curve ratio rather than its shape. The two curves carry different absolute scales, so the ratio was 0.02-0.06 on one structure and 8-12 on another and the returned ``sigma_A`` was flat at 0.15-0.35 regardless of resolution; each curve is now divided by its geometric mean first. Per-shell factors are gauge in the rotation function's correlation, so its placements are unchanged (30/30)
- The fast translation function scores the covariance of two normalised intensities -- the rotation function's LERF1 coefficient ``cw (E_obs^2 - 1) w sigma_A^2`` against the candidate's ``|E_calc(h, t)|^2``, normalised per candidate by the same Wilson fit -- instead of a raw-``|F_calc|^2`` ratio that was not a correlation and, on the four largest panel structures, was higher 40 A from the true position than at it. One FFT on a grid a third of the set's resolution apart with parabolic peak refinement replaces the 16-point coarse grid and three 100-point local refines; the Rice/Woolfson likelihood at a fixed Luzzati ``sigma_A`` picks among the top peaks and ranks the candidates. 30/30 true poses at the default window and 30/30 with the window removed, against 18/30 before
- The translation stage runs in the configured float and complex dtypes rather than hard-coded double
- Removed ``use_llg_tf``, ``n_translation_peaks`` and ``translation_grid_steps`` from the pipeline; the likelihood always picks the translation, and the grid is sized by resolution
- Removed the per-candidate ``SigmaAEstimator`` fit from the placement loop. The likelihood that ranks candidates uses one ``sigma_A`` for all of them, so no candidate is scored against a model error fitted to itself
- The translation search defaults to the rotation search's resolution window instead of all data. With all data it placed the four largest panel structures (2DQ6, 3VRJ, 4BX9, 6G9X) at the right orientation and 20-56 A from the true position on every trial, and its own score was higher at the wrong place than at the deposited pose; the benchmark had only ever checked the rotation. 30/30 true poses within 0.32 A against 18/30, at roughly a fifth of the wall clock
- The P1 copy each rotation candidate is evaluated through is gridded at two thirds of the translation window's resolution rather than at the model's default 1.0 A. Coherence with the fine grid 0.9995 or better on all four structures measured; 10-38 ms per candidate against 200-860
- The pose-recovery harness compares against Cartesian symmetry mates and checks the translation. It compared a Cartesian rotation against fractional symmetry matrices, so in trigonal and hexagonal cells two of six and four of twelve correct mates read as 30 and 21 degrees; every recorded 2DQ6 failure was one of them
- Repaired the alignment integration tests, which imported a deleted module and passed removed arguments
- The alignment package uses the shared FFT-size and Euler-matrix helpers instead of its own copies, and drops four unused rotation utilities and two duplicated symmetry helpers. Bit-identical placements on the benchmark panel
- The translation likelihood's model error comes from the shared ``SigmaAEstimator`` instead of a local 81-point scan over every reflection. It returns ``alpha`` and ``beta`` per reflection rather than a per-shell ``sigma_A``, so the likelihood no longer assumes ``<E_calc^2>`` is exactly one. Outcome-neutral over 70 seeded cells, zero flips; not measurably faster
- Fixed the translation likelihood's variance convention, which scored acentric reflections at twice the variance intended -- 90-95% of reflections. The alignment package carried its own Rice and Woolfson parameterised by the *amplitude* variance and handed both branches the same number, where the acentric branch needs half what the centric one does. It now uses ``base.targets.xray_likelihoods.rice_per_refl``, which takes the complex variance and derives the centric case from it
- Removed ``experimental/alignment/distributions.py``. Its ``stable_log_bessel_i0`` also carried a wrong asymptotic coefficient, giving -2.9e-5 at x = 50 against -5.4e-7 for the correct term; the shared implementation uses ``log(i0e(z)) + z``, which is exact
- Molecular-replacement candidates are ranked by the translation function's likelihood, not by an analytical-scale R-factor. 30/30 on the ten-structure panel against 29/30. Over a ten-seed sweep the two tie at 36/40 and the likelihood's advantage is placement rather than count -- median residual 1.43 deg against 1.62, carried by one structure. Selectable through ``rank_by``; the correlation is the third option and is the worst of the three at 32/40, despite a rank-level harness rating it best on a truth label that disagrees with coordinate superposition
- The likelihood ranking costs about 2.5x the wall clock, from a per-candidate sigma_A fit
- The molecular-replacement pipeline's ``verbose`` levels are a documented contract routed through one emitter, rather than ``if verbose > 0: print(...)`` at seventeen sites. Level 2 emits one machine-readable ``CAND`` line per rotation candidate carrying every score the selection could have used, so a wrong placement can be diagnosed from the run itself instead of from a harness that re-implements the placement loop and then disagrees with it
- The molecular-replacement pipeline places all ``n_rotation_candidates`` (now 25, was 15) and returns the best, instead of stopping once a placement beat an R-factor threshold. The old rule made the answer depend on the order the rotation function happened to produce and could accept the third candidate without scoring the tenth. Measured neutral -- identical placements on 10 structures x 3 seeds and on a 10-seed sweep of the marginal cases -- at about 1.6x the wall clock
- The Wilson normaliser converges in 8-12 IRLS iterations instead of 26-102. Its stopping rule was ``|dL|`` per reflection against 1e-10, which asks eleven significant digits of a normalisation curve; it is now relative to the improvement so far, which is scale-invariant for the same reason the absolute form was chosen
- ``<E^2> = 1`` is solved in closed form for the intercept, so the identity no longer degrades as the convergence tolerance is loosened
- The Wilson fit runs in the configured float dtype rather than hardcoded double, and builds its normal-equations matrix once -- it is constant for a Gamma with a log link. Identical placements on all 30 benchmark cells, at 1.8x the speed; the unit suite went from 968s to 485s
- Molecular replacement is now a rotation search feeding a translation search and nothing else; the pipeline returns a placement and stops. End-to-end pose recovery over 10 structures x 3 seeds went 18/30 to 30/30, at about a sixth of the wall clock
- Removed the ML rescore from between the two searches. It reordered a shortlist that already contained the answer, and cost 6 of 30 placements
- Removed the post-placement dense rotation re-sampling and rigid-body polish. They refined a correct placement away from truth on 2DQ6, 3GR5 and 4BX9; refining a placement is downstream refinement's job
- The translation search weights reflections by inverse variance, which it previously did not do at all, and both searches normalise through one shared Wilson fit built once per run instead of five private ones. This is what recovered 6G9X
- The translation search's resolution window is a parameter (``tf_d_min``/``tf_d_max``) rather than a docstring; it defaults to the existing behaviour of no cut
- The alignment package takes its device from the configured default throughout, instead of reading it off whichever model or tensor was nearest
- Removed the alignment package's unreachable modules: quaternion transforms, a second Wigner implementation, clash scoring, vector sampling, the Lattman-Love interpolator, the E-value convention layer and its French-Wilson posterior, and the unused half of the spherical-harmonic expansion
- Fixed the reciprocal-space symmetry convention in the alignment package (``h.S``, not ``S.h``)
- Fixed ``hkl_symops_to_cartesian`` returning non-rotations in trigonal and hexagonal settings, which corrupted the anisotropy projection
- Fixed the overall-anisotropy fit, which regressed log intensities with no constant term and so absorbed the ``-gamma`` offset into the tensor
- Fixed molecular-replacement rotation candidates being composed onto each other instead of onto the search model
- Fixed assigning a ``SpaceGroup`` object to ``Model.spacegroup`` being a silent no-op that then made the correct name assignment raise
- The ML rescore can score orientations by weighted least squares on E-space intensities (``target='wls'``) instead of the Rice/Woolfson likelihood. Measured indistinguishable from the Rice over 10 structures x 10 seeds; both remain worse than not rescoring at all
- The rotation function estimates ``sigma_A`` from the data instead of assuming it. Total scattering per shell is rotation-invariant, so ``Sigma_obs(s)/Sigma_calc(s)`` measures the model's resolution-dependent deficiency before placement; it replaces the Luzzati falloff from an estimated coordinate error and the Babinet bulk-solvent term with its two universal constants
- Removed the relative Wilson-B match from the rotation search. It multiplied ``F_calc`` by a smooth function of ``|s|`` that the normalisation then divided straight back out
- Added ``torchref.scaling.weighting``: measurement and model error combined as one inverse-variance weight per reflection, restoring the observed-side weighting that left with the French-Wilson posterior
- ``apply_shell_variance_weights`` is off by default in the rotation function. It is a per-shell weight, and per-shell weights are absorbed by the correlation; switching it on moves nothing
- The rotation function, ML rescore and translation search now normalise through the shared Wilson normaliser by default, replacing the French-Wilson posterior on the observed side. Rank-neutral over 10 structures x 10 seeds; the rotation function is about twice as fast, since the posterior and its D-factor iteration are no longer on the default path
- The observed-side ``DFAC`` weighting went with it. Weighting is a separate concern from scaling and is being rebuilt as its own object; until then the rotation function applies no measurement-error weight
- Added ``torchref.scaling.WilsonNormaliser``: an absolute normaliser that fits ``Sigma(s)`` as a Gamma GLM with a log link and divides it out, so ``<E^2> = 1`` holds as an identity of the fit rather than as a separate normalisation step
- Extracted the Chebyshev resolution basis into ``torchref.scaling.basis``, shared with the isotropic scale, and gave it an explicit range so a curve fitted on one reflection set can be evaluated on another
- Moved epsilon onto ``SpaceGroup.epsilon(hkl, friedel=)``; the alignment package's own copy disagreed with it in trigonal and hexagonal groups and dropped the centring coset. The default keeps the Friedel-folded count sigma_A is calibrated against, and the molecular-replacement likelihood asks for the conventional one
- The rotation function and the ML rescore take their E-value convention as a class, so the observed and calculated sides are normalised by one rule rather than by nine private converters
- The ML rescore now receives the observation sigmas, which the rotation function computed the French-Wilson posterior from and then discarded
- Removed the rescore's ``scat_mode``, a second knob for the decision the E convention already makes
- The Sim rescore and the three translation-search sites take their E values from the convention too, so the alignment package has one normaliser rather than nine
- Removed ``wilson_normalise`` and ``wilson_normalise_epsilon``, which the convention replaced
- Fixed the rotation search, placement pipeline and ``align`` reading ``model.initialized``, which moved to ``model.ctx``; the tests covering it are slow-marked, so the break was invisible to a default test run
- Replaced the fast rotation function's keyword surface with ``rotation_search(model, data, model_error_A)``; the caller's coordinate error is now used rather than overwritten by an estimate from the atom count
- Removed the rotation function's dead modules, engine variants, debug environment switches and unreachable knobs
- ``Model``'s iso/aniso partition is now derived on access instead of being rebuilt eagerly, so a copy cannot inherit a stale one
- The rotation function's Wigner small-d blocks are memoised, so a process running more than one search builds them once
- The rotation function now takes its working precision from ``dtypes.float`` and its device from ``resolve_device``, instead of hardcoding float64 and reading one input's device
- The rotation function's spherical-Bessel recurrence rescales by a power of two as it runs, so the ladder no longer needs float64's exponent range
- The rotation function's Wigner eigendecomposition and anisotropy fit moved to the host, so neither requires float64 on the accelerator
- Removed the rotation function's duplicate Euler, Rodrigues and reciprocal-symmetry helpers in favour of the shared primitives
- Removed the rotation function's redundant calc-side resolution mask and its second bandwidth/resolution coupling call
- ``bessel_sh_expand`` lost its unread ``chunk_size`` argument and ``french_wilson_preprocess`` its unread ``sqrt_mean_F2`` output
- The rotation function's observed-side chain (French-Wilson, LERF1, shell variance weights, relative Wilson B) now runs once per unique reflection instead of once per symmetry copy; only the geometry is unrolled
- The rotation function assigns resolution shells once and shares them, instead of the Wilson normalisation and the variance reweight deriving edges that disagreed at the shell boundaries
- The rotation function masks observations to the bandwidth-coupled resolution before the symmetry unroll rather than after
- The rotation function warns on Bijvoet-unmerged data, whose shared canonical index would weight those reflections twice
- The rotation function no longer concatenates the antipodal copy onto either reflection set: only even harmonic degrees are computed, for which it is an exact factor of two, so it scaled the rotation function by four and changed no ranking. Raw ``RotationPeak.score`` and ``RotationSolutions.scores`` are therefore a quarter of their previous values; z-scores are unchanged
- Kept the rotation function's relative Wilson-B fit: knocking it out was measured rank-neutral but worth only 2% of the runtime once the fit moved to the unique reflection set
- Added ``supports_double`` / ``widest_float_dtype`` / ``widest_complex_dtype``: where precision is load-bearing the width now comes from the device rather than a hardcoded ``float64``, so a backend without it gets the working dtype instead of an error
Version 0.7.0
----------
- Fixed cif reading bug discarding new mmCIF field for aniso ADPs 
- Removed the stored real-space coordinate grid; ``build_electron_density`` takes a grid shape and device, and ``ModelFT.real_space_grid()`` builds one on demand
- Fixed ``ModelFT`` restore dropping a node-field ADP representation, and added the anisotropic ``field_aniso`` case; both models now share one wrapper-rebuild path
- Fixed the node load and node smoothness restraints being inert in ``field_aniso`` mode
- ``create_from_state_dict`` now restores on CPU and moves only when passed a device; it previously left three of the four parameter wrappers on CPU while claiming the default device
- Separated model configuration and provenance into ``ModelContext``. It now holds the unit cell, space group, atom table, link records, hydrogen settings, and input paths.
- Refactored ``Symmetry`` as a crystallography-free class with transform primitives, and made ``SpaceGroup`` a specialised subclass.
- Moved geometry predicates, HKL verbs, and grid-size helpers onto these classes as methods.
- Rebuilt geometry restraints from the topology instead of intra-residue builders. ``torchref.restraints`` was removed, restraint dictionaries are now plain nested dicts, and residues are identified by ``(chain, resseq, icode)`` to fix insertion-code merging.
- Reworked hydrogen generation as template instantiation over the topology. ``Model.hydrogenate`` now aligns monomer templates onto heavy atoms present, generation is the default, and ``AtomGraph.exclusions_12_13_14`` derives non-bonded exclusions from bond connectivity.
- Added ``Topology`` as a ``ResidueGraph`` over an ``AtomGraph`` with typed edge blocks and ``subset`` / ``copy`` operations that reindex surviving edges.
- Made ``HydrogenTopology`` a dataclass, changed ``Symmetry`` classes to dataclasses over ``DeviceMixin`` instead of ``nn.Module``, and removed unused ``Cell`` gradient plumbing and the ``ReciprocalSymmetryGrid`` / module-level expansion functions.


Version 0.6.4
----------
- Fixed the bulk-solvent ``F_sol`` staying at the starting model's mask for every refinement macrocycle
- Fixed restraint dictionaries defining several compounds yielding restraints for only one of them
- Fixed chirality restraints being dropped for the ``positiv``/``negativ`` spellings used by the CCP4 library
- Compounds that come back with no bond restraints are now reported
- Fixed written phases (``PH-model``, ``PHWT``, ``PHDELWT``) being negated for reflections whose input Miller indices lay outside the CCP4 ASU
- Reflections remapped to the CCP4 ASU on load are now reported
- Switched the scaler's default scale-fit objective from ``nll`` to unit-weight ``ls``
- Replaced the per-bin ``log_scale`` with a Chebyshev polynomial ``c_iso`` in sin(theta)/lambda
- Replaced the solvent Debye-Waller factor with ``k_sol exp(-ln2 (ss/ss_half)^n)``, merged sigmoid exponential form
- Fixed the solvent-mask candidate enumeration, which missed voxels near the atom's grid node
- Removed the solvent-mask Gaussian smoothing
- Fixed the scale fit's float64 normalisation constant, which broke MPS
- Batched direct summation now returns ``dtypes.complex`` instead of always ``complex128``


Version 0.6.3
-------------
- Fixed peptide-linked residues keeping their free-amino-acid restraint angles
- Added switch to turn off caching mixin
- Switched the ADP distribution restraint from a Gaussian in log(B) to the shifted inverse-gamma distribution of Masmaliyeva & Murshudov (2019)
- Reworked outlier rejection to follow wilson criteria
- Reworked free flag generation so Friedel pairs get matching flags
- Fixed breaking bug were sigma_A estimation did not work on mps 


Version 0.6.2
-------------
- Deprecated outlier flagging strategy 
- Rewrote X‑ray targets into five independent classes (nll, nll_beta, ml, ml_noalpha, ml_full) and XRAY_TARGETS.by_name.
- Consolidated X‑ray loss math into Gaussian, Rice, marginalised Rice primitives; NLLXrayTarget replaces GaussianXrayTarget.
- Fixed crash in difference refinement with mismatched reflection files: HKL reindexing (``validate_hkl``/``remap``/``reduce_to_spacegroup``) now carries all per-reflection fields (including the anomalous bookkeeping read by ``hkl_for_sf()``) instead of a hardcoded subset.
- Added a metal shader for structure factor calculation
- Standardized structure factor calculation geometry
- Split gpu tests into cuda and mps
- Reworked VDW pair list creation
- Reworked backend dispatch: which kernel runs is now read from one declarative table per kernel family (device, dtype, availability probe, failure policy), replacing two hand-written if/elif ladders.
- Added preconditioned L-BFGS optimizer for joined refinement.


Version 0.6.1
-------------
- Fixed bug in beta estimation that caused instability in GPU refinement
- Fixed reporting bug in collection scaler were rfactor reporting would ignore masks


Version 0.6.0
-------------
- Switched to per-atom cutoff radii for the electron-density sampling and added a global sigma cutoff
- Implemented Phenix style sigma A weighting in the Maximum likelihood target (New default for refinement)
- Set Ramachandran restraints to be off by default (to enable specify a non zero weight)
- Added collection versions of the sigma A target
- Fixed bug in Rfree-generation now min 1000 reflections per bin, min 50 free reflections max 2% and 10 bins
- Deprecated internal coordinates
- Renamed Maximum likelihood target to Rice target, Sigma A target to Maximum likelihood target.
- Moved alignment into experimental
- Fixed Fast rotation function, current blocker on alignment is the rescoring function
- Moved kinetic to experimental 
- Added monolithic refinement under experimental
- Moved ensemble refinement to experimental
- Fixed antechamber handling of non-standard residues
- Refinement now freezes all residues missing restraints (xyz)
- Changed reflection data accessor api to use property style data selection
- Added benchmark to the paper folder where we refine from alphafold start coordinates
- Added isotropic / anisotropic switch and selection to refinement cli
- Updated many docstrings, and fixed some bugs

Version 0.5.3.3
---------------
- Fixed U_aniso parametrization and line search instability during refinement with anisotropic b-factor
- Fixed kinetic module import 
- Set default similarity weight in difference refinement to 0

Version 0.5.3.2
---------------
- Added 10GB Gram requirement for default gpu device selection
- Slaved cli device detection to the default device selection
- Fixed device mismatch crash on CUDA/MPS when the VDW pair list was refreshed mid-refinement: the maintenance-triggered rebuild now migrates the fresh VDW pair list, hydrogen topology, and exclusion hash to the model device (PR #19)

Version 0.5.3.1
---------------
- Fixed problem where TorchRef defaults to old gpus and crashes, now checking if gpu is actually usable, before setting default device to cuda, if not it will default to cpu and print a warning.

Version 0.5.3
-------------
- Fixed dtype inconsistencies and centralized dtype handling
- Centralized default device handling
- Added compatability with Metal performance shaders

Version 0.5.2
-------------
- Cleaned up build solvent mask calculation
- Added DeviceMixin for centralizing device handling accross all classes

Version 0.5.1
-------------
- Fixed masked tensor problem under torch 2.9
- Added Link parsing and restraint as bond
- Reduced memory usage during neighbor search for VDW target
- Separated out loss functions from targets, logic moved to base/targets
- Added Triton kernels with analytic backward for all four xray Targets and most other Targets
- Cached XrayTarget.get_data constants across closures
- Replaced slow tensor[indices] backwards (sort + dedup scatter) with ``index_add_`` in the symmetry extractor, scaler bin gathers, and MixedTensor; skip the indexing in get_iso / get_aniso when it covers all atoms

Version 0.5.0
-------------
- Fixed two bugs in restraints related to peptide bonds
- Centralized closure infrastructure in Lossstate, added systematic parameter freezing and cache management to the optimization functions as well as step pruning
- Switched main xray target to bhattacharyya distance between observed and calculated structure factor distributions.
- Added model error estimation based on bfactor distribution and fischer information
- Fixed weighting necessity by moving to unscaled log likelihoods for all targetes, overfitting weights remain
- Fixed missing angle in Proline geometry
- Fixed restraint issues where peptide bonds were not being recognized accross altlocs
- Fixed OOM errors in refinement caused by solvent map creation by explicitely handling symmetry
- Implemented VDW restraints between symmetry mates, and vectorized spatial hashing for gpu friendly neighbor search
- Migrated difference-refine script to the collection infrastructure, similarly migrated validate_ded and phased difference map
- Added free CC calculation to validation_ded in reciprocal space
- Deprecated scaler cli args as they are now always scaled together
- Merged collection and basic kinetic infrastructure
- Renamed column names in difference-refine output to be more concise and accurate
- Fixed some bugs in collection architecture and 
- Added PDB deposition headers (REMARK 3 with refinement statistics) and mmCIF coordinate writing
- Added unified RefinementMetadata that renders to both PDB and mmCIF using PDBx/wwPDB field names
- All refinement CLI scripts now write both PDB and mmCIF by default
- Added torchref.add-metadata CLI tool for adding metadata to existing files
- Added input file header pass-through for PDB and mmCIF

Version 0.4.3
-------------

- Refactored and standardised cli args
- Unified validation-ded and difference-refine scaler logic and added flags for separate vs shared scalers
- Added option for other difference targets mainly rice, Does not seem to make a difference 

Version 0.4.2
-------------

- Fixed bug where reflection data object in the refinement was not created on cuda when specified. 
- Fixed macos crash, not catching compilation error in c++ extension for scatter add

Version 0.4.1
-------------

- fixed weird pytorch numpy compatability issues

Version 0.4.0
-------------

- Added cli tool for running difference refinement
- Refactored targets
- Cleaned up and refactored dispatch in Structure factor calculation
- Added fast cpu scatter c++ implementation for structure factor calculation
- Added 2 custom triton kernels for structure factor calculation, one for the general case and one optimized for the common case of isotropic B factors and orthogonal cells
- Added difference target
- Added dataset scaling to DatasetCollection
- Added Ramachandran restraints
- Added partial compilation support for loss states
- Added map module for calculating and writing maps
- Added real space targets for refinement
- Finetuned default hyperparameters for LBFGSRefinement
- Switched from downloaded entire monomer library to lazy downloading of required monomers

- Added basic Langevin thermostat based SA refinement implementation, needs testing and validation
- Added 2 dev implementations of internal coordinate parametrisations, need more testing
- Added a Amber loss target function, needs valdiation


Version 0.3.2
-------------

- Hotfix for imports, bundled data and a minor bug in LBFGSRefinement

Version 0.3.0
-------------

- Initial public release

Major changes: 

- Separated ED building and SF calculations out of the ModelFT module 
- Resolved scaler model dependency
- Restraints are now tracked by the base model object instead of refinement
- Targets are no longer dependent on the refinement object, only reference the respective model and if required ReflectionData object
- Restructured math functions and renamed to base 
- Introduced centralized dtype handling for all functions to avoid conflicts, default dtypes are torch.float32, torch.int32, and torch.complex64
- Unified base Symmetry and SpaceGroup object into the SpaceGroup object

Additions

- Implemented Fast translation and rotation search functions for alignment module (rotation search does not work reliably at the moment)
- Implemented Rigidbody refinement
- Implemented and validated scaffold for full stack alignement (does not work reliably)
- Added E factor conversion to the reflection_data class

Version 0.2.0
-------------

- Core refinement framework
- Support for MTZ, PDB, and CIF file formats
- Geometry restraints (bonds, angles, torsions, planes)
- Bulk solvent model
- GPU acceleration via CUDA
- Full state_dict support for checkpointing

Version 0.1.0
-------------

*Internal release*

- Initial implementation
- Basic structure factor calculations
- Least squares target function
