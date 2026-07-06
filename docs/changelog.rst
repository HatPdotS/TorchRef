Changelog
=========

Version 0.6.1
-------------
- Fixed bug in beta estimation that caused instability in GPU refinement


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
-------------
- Fixed U_aniso parametrization and line search instability during refinement with anisotropic b-factor
- Fixed kinetic module import 
- Set default similarity weight in difference refinement to 0

Version 0.5.3.2
-------------
- Added 10GB Gram requirement for default gpu device selection
- Slaved cli device detection to the default device selection
- Fixed device mismatch crash on CUDA/MPS when the VDW pair list was refreshed mid-refinement: the maintenance-triggered rebuild now migrates the fresh VDW pair list, hydrogen topology, and exclusion hash to the model device (PR #19)

Version 0.5.3.1
-------------
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
- Replaced slow tensor[indices] backwards (sort + dedup scatter) with index_add_ in the symmetry extractor, scaler bin gathers, and MixedTensor; skip the indexing in get_iso / get_aniso when it covers all atoms

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
