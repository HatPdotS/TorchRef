Changelog
=========


Version 0.7.0
-------------

Breaking changes
~~~~~~~~~~~~~~~~
- Removed ``torchref.restraints``. Restraints are built from the new topology graph and live in ``torchref.topology`` (``restraints``, ``builders``, ``matchers``, ``nonbonded``, ``ramachandran``, ``riding``, ``monomer.cif``/``library``/``modifications``); restraint dictionaries are plain nested dicts and residues are keyed by ``(chain, resseq, icode)``
- numba is no longer a dependency; the per-residue restraint matchers it compiled are faster as plain Python
- Hydrogens present in the input are now kept (``strip_H`` defaults to ``False``, was ``True``) and enter the structure factors (``hydrogens_in_xray``, default on). Generating missing hydrogens stays opt-in: ``torchref.refine --add-hydrogens`` or ``add_hydrogens=True``. ``exclude_H_from_sf`` is a deprecated inverted alias of ``hydrogens_in_xray``
- ``torchref.phased-difference-map`` is renamed ``torchref.difference-map`` (the old name remains as an alias). It defaults to the weighted amplitude difference on the dark model's phases, the map ``torchref.validate-ded`` scores, and ``-lm``/``--light-model`` is optional
- The difference MTZ is reduced from 33 columns (46 under ``--two-moment``) to 17 standard CCP4 names grouped into named datasets (``observed``, ``difference``, ``light_model``, ``extrapolated_light``, ``two_moment``); the alternative constructions are behind ``--all-columns``. ``DELFWT`` is no longer written: the file carries unweighted ``DF``/``SIGDF`` on ``PHDELWT`` with weight columns ``W_IVW``/``W_SD`` and the scale ``KSCALE``, so build the map with ``torchref.mtz2map -csf DF -cw W_IVW -cphi PHDELWT``
- Renamed difference MTZ columns: ``DFc_complex`` is ``DFc_phased``, and ``Fextp``/``Fextc``/``Fextb`` are ``FEXT_PHASED``/``FEXT_SCALAR``/``FEXT``
- Refinement output no longer copies the input file's refinement header. Crystal, sample and chemistry records are carried through; ``REMARK 2``/``3``/``500``, ``AUTHOR`` and ``JRNL`` are replaced by this run's own, and the mmCIF ``_refine`` items are no longer inherited. Removed ``pdb.write(template=)`` and ``custom_remarks``
- Model configuration and provenance (cell, space group, atom table, links, hydrogen settings, input paths) moved into ``ModelContext`` at ``model.ctx``; e.g. ``model.initialized`` is ``model.ctx.initialized``
- ``Symmetry`` is a crystallography-free dataclass with ``SpaceGroup`` as its subclass, and geometry predicates, HKL operations and grid-size helpers are methods on them. Removed ``ReciprocalSymmetryGrid``, ``torchref.symmetry.grid_utils`` and the ``Cell`` gradient plumbing
- ``ModelFT`` no longer stores a real-space coordinate grid: ``build_electron_density`` takes a grid shape and device, and ``ModelFT.real_space_grid()`` builds one on demand
- ``ReflectionData`` no longer holds scale parameters, scale fitting or E-value conversion; use ``WilsonNormaliser`` for E values and the observation attributes and subset views instead of the deprecated getters or ``data()``
- ``DatasetCollection.scale()`` fits all datasets jointly with ``DatasetScaler`` (a per-dataset log scale and anisotropy, centred over datasets so no dataset is the reference, on an inverse-variance-weighted consensus amplitude). Members become ``ScaledDataset`` views carrying live corrections, with the raw values as ``F_raw``/``I_raw``
- ``create_from_state_dict`` restores on CPU and moves only when passed a device
- Rigid-body refinement no longer refines the scale inside the rigid-body L-BFGS; ``refine_scaler`` fits it between cutoffs
- ``ModelCollection`` stores populations as one shared activation fraction plus a per-timepoint branching; freezing fractions is collection-wide, and independent populations use ``set_fraction_override``
- Removed ``CollectionRiceTarget`` (use the ``ml`` row of ``COLLECTION_XRAY_TARGETS``), renamed the kinetic ``xray_weight_rice``/``xray/rice`` weight to ``xray_weight_ml``/``xray/ml``

Fixes that change results
~~~~~~~~~~~~~~~~~~~~~~~~~
- Fixed anisotropic ADPs being dropped from mmCIF files that store them in the standard ``_atom_site_anisotrop`` loop (every PDB and PDB-REDO mmCIF), which loaded every atom as isotropic
- Fixed the empirical-Bayes extrapolation over-weighting the dark-state variance by ``1/(1-f)^2`` (1.64x at f = 0.22), which over-shrank every reflection; the default extrapolated map changes
- Fixed ``DatasetCollection.scale`` fitting the inter-dataset scale on the free reflections as well as the work set
- Fixed ``refine_rigid_body`` leaving the caller's reflection data truncated to the last cutoff for the rest of the run
- Fixed the VDW pair search missing pairs in oblique cells, whose grid cells were narrower than the cutoff (up to 0.23 % more pairs)

Other fixes
~~~~~~~~~~~
- Fixed mmCIF loop cells being written unquoted, which split values containing whitespace on read-back
- Fixed PDB coordinate and B-factor columns dropping trailing zeros
- Fixed loss aggregation ignoring the configured floating-point dtype
- Fixed assigning a ``SpaceGroup`` object to ``Model.spacegroup`` being a silent no-op
- Fixed ``f_sol_override`` overwriting the scaler's cached ``F_sol``, so a later call without an override read the wrong solvent
- Fixed ``torchref.difference-refine`` crashing at ``--verbose 0``
- Fixed ``paper/make_ded_maps.py`` pairing ``WDF`` with ``PHIC_diff``
- ``create_from_state_dict`` no longer leaves three of the four parameter wrappers on CPU while reporting the default device, and ``ModelFT`` restores ``cif_path``

New features
~~~~~~~~~~~~
- Riding hydrogens: ``Model.set_hydrogen_mode("riding" | "free")`` / ``Refinement.set_hydrogen_mode`` refine only heavy atoms through ``RidingXYZTensor``, with refinable methyl/hydroxyl torsions and water orientations. Hydrogen generation instantiates monomer templates over the topology and reads the user's restraint CIF (``cif_path``, ``torchref.refine --cif``); it respects ammonium nitrogen types and leaves linked hetero atoms (acetyl caps, Schiff bases, glycosylated ASN, metal-bound HIS) without displaced hydrogens
- ``torchref.refine --add-hydrogens`` and ``--hydrogens-in-xray/--no-hydrogens-in-xray``
- AMBER targets consume TorchRef-owned coordinates, including hydrogens and riding-orientation gradients, through a validated atom map
- Node-field ADP representation: ``set_adp_mode`` / ``torchref.refine --adp-mode`` gain ``field``, ``field_aniso`` and ``preserve`` (keeps the loaded ADPs untouched), with ``--adp-mode-set``, ``--adp-nodes`` and ``--reflections-per-adp-parameter``, and node load and magnitude restraints
- ``Topology``: a residue graph over an atom graph with ``subset``/``copy``; ``AtomGraph`` carries CCP4 energy types, and mmCIF models read their ``_struct_conn`` links
- Two-moment intensity target (``CollectionTwoMomentIntensityTarget``) for crystal-to-crystal spread in activation: ``torchref.difference-refine --two-moment``/``--lambda-twin``/``--refine-lambda-twin``
- ``sigma_D`` difference-power estimator (``torchref.refinement.model_error_estimation.sigma_d``) and the ``difference_sd`` collection target (``--difference-target difference_sd``)
- ``--ded-weight {inverse_variance,sigma_d,none}`` (default ``inverse_variance``) and ``--sigma-d-gamma`` on ``torchref.difference-map``, ``torchref.difference-refine`` and ``torchref.validate-ded``; ``validate-ded`` reports correlations for every weighting side by side and records ``mask_source``
- ``torchref.mtz2map`` gains ``--column-weight``/``-cw``, ``--column-scale``/``-ck`` and ``--units {sigma,electrons,raw}`` for maps in e/A^3; ``-n``/``--normalize`` is deprecated
- ``torchref.simulate-noisy-data`` and ``FcalcDataset.add_noise``: simulate merged intensities and report R-split and CC between half-datasets
- CrystFEL ``partialator`` ``.hkl`` reading via ``ReflectionData.load_crystfel_hkl``
- ``--xray-mode nll_i`` (Gaussian on intensities), an ``observable`` axis on the X-ray target taxonomy, and ``COLLECTION_XRAY_TARGETS``
- ``--output-remarks`` for author header text. The refinement header records target, optimizer, ADP model and scale target, the free-set source and seed, and the starting model by file name; mmCIF keeps the ``_software`` chain of prior refinements
- ``WilsonNormaliser`` (absolute normalisation as a Gamma GLM), ``torchref.scaling.basis`` (shared Chebyshev basis) and ``torchref.scaling.weighting``
- ``SpaceGroup.epsilon(hkl, friedel=)``, ``ScalerBase.multiplicative_scale()``, ``supports_double``/``widest_float_dtype``/``widest_complex_dtype``
- Batched collection evaluation: ``compute_component_fcalcs``/``mix_component_fcalcs``, ``DatasetCollection.component_structure_factors``, ``CollectionScaler.forward_batched``, ``stack_F_obs``/``stack_I_obs``/``stack_masks``, scaled ``I``/``sigI`` views

Performance
~~~~~~~~~~~
- The VDW pair search uses a k-d tree on CPU (5BOV with hydrogens 44 s to 0.6 s), as does the ADP-locality neighbour list (4-8x)
- Switching to riding hydrogens is no longer quadratic in atom count (4BX9 132 s to 1.5 s)
- Restraint building skips the pair search it discards when adding hydrogens, and no longer pays numba compilation (13.5 s cold)
- Rigid-body angles are preconditioned by the radius of gyration, so the step converges in about half the gradient evaluations
- Grid sizing is lazy and cached

Experimental: molecular replacement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- The pipeline is a rotation search feeding a translation search, and returns the best placement of 10 rotation candidates ranked by the translation likelihood (``rank_by``). Pose recovery on the 10-structure x 3-seed panel went from 18/30 to 30/30
- ``rotation_search(model, data, model_error_A)`` replaces the rotation function's keyword surface; peaks exclude symmetry mates and raw scores are a quarter of their previous values (z-scores unchanged)
- The fast translation function scores a normalised intensity covariance on one FFT grid, weighted by inverse variance, within the rotation search's resolution window (``tf_d_min``/``tf_d_max``)
- Both searches normalise through the shared ``WilsonNormaliser`` and estimate ``sigma_A`` from the data
- The SH-Bessel expansion is about 8x faster at L=100, with a fused C++ Legendre kernel
- The alignment package runs on MPS without float64
- Removed the ML rescore, post-placement re-sampling and rigid-body polish, ``DirectModelEvaluator``, ``use_llg_tf``, ``n_translation_peaks``, ``translation_grid_steps``, ``wilson_normalise``/``wilson_normalise_epsilon``, and the unreachable modules (``ball_transform``, ``clashscore``, ``distributions``, ``jax_subpixel_peaks``, ``rigid_body``, ``sampling``, ``transform``)
- Fixed the reciprocal-space symmetry convention (``h.S``, not ``S.h``), rotation candidates being composed onto each other, the translation likelihood scoring acentric reflections at twice the intended variance, the overall-anisotropy fit having no constant term, and the rotation function contracting unconjugated coefficients on MPS

Internal
~~~~~~~~
- Reorganised the test suite by API ownership and fixture scope, moved extra structures to the slow tier, and pinned the pair-search tests to CPU
- Pull requests into ``dev`` run one CPU and one MPS test job
- Every hard-coded dtype is either configured or carries a ``# dtype-ok:`` justification
- Collection targets share the single-dataset ``_loss_inputs``/``_per_refl`` seam; the amplitude and intensity Gaussians are one implementation


Version 0.6.4
-------------
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
