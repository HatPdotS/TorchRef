Changelog
=========

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
