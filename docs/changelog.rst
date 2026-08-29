Changelog
=========


Unreleased
----------
- Fixed the reciprocal-space symmetry convention in the alignment package (``h.S``, not ``S.h``)
- Fixed ``hkl_symops_to_cartesian`` returning non-rotations in trigonal and hexagonal settings, which corrupted the anisotropy projection
- Fixed the overall-anisotropy fit, which regressed log intensities with no constant term and so absorbed the ``-gamma`` offset into the tensor
- Fixed molecular-replacement rotation candidates being composed onto each other instead of onto the search model
- Fixed assigning a ``SpaceGroup`` object to ``Model.spacegroup`` being a silent no-op that then made the correct name assignment raise
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
