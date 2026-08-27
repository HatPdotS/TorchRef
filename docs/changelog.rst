Changelog
=========


Unreleased
----------
- Added ``Symmetry``, a crystallography-free symmetry group carrying the operations and every verb derived from them; ``SpaceGroup`` now specialises it
- ``Symmetry`` exposes the transform primitives ``apply_rotations`` / ``apply_translations`` / ``phase_factors`` and a cached ``reciprocal`` stack, replacing seven separate spellings of ``R^T h``
- Moved the centric, systematic-absence and epsilon predicates onto ``Symmetry``; ``is_centric_from_hkl`` and ``get_centric_acentric_masks`` are gone
- Moved the HKL asymmetric-unit verbs onto ``SpaceGroup`` as ``expand_hkl`` / ``reduce_hkl`` / ``complete_hkl`` / ``canonicalize_hkl``; the module-level functions are gone
- Moved the grid-size helpers onto ``Symmetry``; removed ``torchref.symmetry.grid_utils`` and the duplicate ``spacegroup`` module-level functions
- Map symmetrization is now ``Symmetry.symmetrize_map``, caching one operator for the most recent grid shape; ``MapSymmetry`` and ``MapSymmetryDirect`` are private
- Symmetry classes are dataclasses over ``DeviceMixin`` instead of ``nn.Module``, so assigning a ``SpaceGroup`` to a model attribute is no longer intercepted by ``nn.Module.__setattr__``
- Removed ``ReciprocalSymmetryGrid``, ``ReciprocalSymmetry``, ``expand_reciprocal_grid``, ``expand_reflections`` and ``extract_structure_factors_with_symmetry``
- Removed the unused ``Cell`` gradient plumbing (``requires_grad`` argument and property, ``detach``) and the ``CellTensor`` alias
- Removed the ``Symmetry`` alias for ``SpaceGroup``; the name is now a distinct class
- Added ``ModelContext``, holding a model's unit cell, space group, atom table, link records and provenance; ``Model.cell`` / ``.spacegroup`` / ``.pdb`` still work and now read through it
- Moved ``Model``'s configuration and provenance onto the context: ``strip_H``, ``verbose``, ``links``, ``altloc_pairs``, ``initialized``, ``exclude_H_from_sf`` and the input paths are reached as ``model.ctx.*``
- ``Model.copy`` and ``ModelFT.copy`` now copy the context in one step, cloning the space group instead of sharing it
- Removed ``Model.symmetry``; use ``Model.spacegroup``
- ``HydrogenTopology`` is a dataclass with optional fields instead of an ``nn.Module`` whose buffers were attached after construction
- Added ``torchref.topology``: a ``Topology`` of a ``ResidueGraph`` over an ``AtomGraph``, holding the model's connectivity as typed edge blocks with a bond adjacency that answers ``neighbors(i)``
- Topology residues are identified by ``(chain, resseq, icode)``, so a residue with an insertion code is no longer merged with the one it was inserted after
- Added ``AtomGraph.exclusions_12_13_14``, deriving non-bonded exclusions from bond connectivity rather than from which angles and torsions the monomer library happens to restrain
- ``Restraints.restraints`` is now a plain nested dict of tensors instead of an accessor object rebuilt on every read; the per-origin indices are views into the topology's contiguous edge blocks
- Restraint groups are laid out in a fixed order, so a rebuild produces the same row order in any process; previously the origins were concatenated in Python ``set`` iteration order
- Fixed ``cat_dict`` doubling every bond, angle and torsion restraint when called more than once
- Geometry restraints are built from the topology, retiring the intra-residue builder calls, the peptide/disulfide/LINK build methods and the ``TensorDict`` restraint storage
- Hydrogen generation is now template instantiation over the topology: ``Model.hydrogenate`` aligns each residue's monomer template onto the heavy atoms present and reads its hydrogens off, and the bond graph sets how many hydrogens a parent may carry
- Fixed hydrogen placement fitting the template over two bond shells, which spans rotatable torsions the model does not share and left 12% of side-chain hydrogens further than 1.5 A from their parent, where they were discarded
- Hydrogens on a centre whose template omits a real substituent -- a peptide-linked backbone nitrogen -- are now built from the bonded neighbours instead of the template frame
- Hydrogens with a free torsion (hydroxyl, thiol, amine, methyl) are identified from bond connectivity and their dihedral is scanned, rather than taken from whatever the library deposited
- Removed ``Model.generate_hydrogens``; ``Model.hydrogenate`` is the single path and no longer takes ``lbfgs_steps`` or ``max_iter``
- Hydrogens are now present by default: ``strip_H`` defaults to False, and hydrogens a file does not carry are generated on load. New ``add_hydrogens`` argument turns generation off while still keeping any the file has
- Generation is decided per parent, so a partially hydrogenated structure is topped up rather than left as deposited
- Fixed the lazily-cached per-atom buffers (``vdw_radii``, ``Z``, the ITC92 coefficients) surviving a load that changes the atom count, which left them sized for the previous atom set
- Riding hydrogens are no longer placed when the model carries real ones, where they acted as phantom atoms in the non-bonded term
- Fixed the hydrogen valence cap counting only heavy neighbours, so a parent that already carried a hydrogen still had budget for another; generation was not idempotent and a save/reload added a spurious second amide hydrogen to every linked residue
- Moved the riding-hydrogen map from ``torchref.restraints.hydrogen_topology`` to ``torchref.topology.riding``, alongside the generation path it is the heavy-atom-only alternative to
- Added ``Topology.subset`` and ``copy``, plus the same on ``EdgeBlock``, ``ResidueGraph`` and ``AtomGraph``: a subset reindexes the surviving edges rather than re-reading the CIFs and re-matching the templates. An edge is dropped as soon as any of its atoms is, and a residue left with no atoms goes along with its links


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
