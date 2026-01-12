"""
Restraints module for crystallographic refinement.

This module provides classes for building and managing geometry restraints
(bonds, angles, torsions, planes, chirals, VDW contacts) from CIF dictionaries.

Classes
-------

RestraintsNew
    Refactored restraints handler using builder pattern (faster, more testable).
ResidueIterator
    Efficient iterator over residues with pre-grouped data.
RestraintBuilder
    Abstract base class for restraint builders.
BondRestraintBuilder
    Builder for bond length restraints.
AngleRestraintBuilder
    Builder for angle restraints.
TorsionRestraintBuilder
    Builder for torsion angle restraints.
PlaneRestraintBuilder
    Builder for planarity restraints.
ChiralRestraintBuilder
    Builder for chiral volume restraints.
InterResidueBondBuilder
    Builder for inter-residue bond restraints (peptide, disulfide).
InterResidueAngleBuilder
    Builder for inter-residue angle restraints.
InterResidueTorsionBuilder
    Builder for inter-residue torsion restraints (phi, psi, omega).
InterResiduePlaneBuilder
    Builder for inter-residue plane restraints.
"""

from torchref.restraints.restraints_new import RestraintsNew as Restraints
from torchref.restraints.builders import (
    ResidueIterator,
    RestraintBuilder,
    BondRestraintBuilder,
    AngleRestraintBuilder,
    TorsionRestraintBuilder,
    PlaneRestraintBuilder,
    ChiralRestraintBuilder,
    InterResidueBondBuilder,
    InterResidueAngleBuilder,
    InterResidueTorsionBuilder,
    InterResiduePlaneBuilder,
)

from torchref import ROOT_TORCHREF
from pathlib import Path
import os

if not os.path.exists(os.path.join(ROOT_TORCHREF,'external_monomer_library')):
    import warnings
    warnings.warn("External monomer library not found in torchref package root. ", ResourceWarning)
    MONOMER_LIB_PATH = None
else:
    MONOMER_LIB_PATH = Path(os.path.join(ROOT_TORCHREF,'external_monomer_library'))



__all__ = [
    'Restraints',
    'RestraintsNew',
    'ResidueIterator',
    'RestraintBuilder',
    'BondRestraintBuilder',
    'AngleRestraintBuilder',
    'TorsionRestraintBuilder',
    'PlaneRestraintBuilder',
    'ChiralRestraintBuilder',
    'InterResidueBondBuilder',
    'InterResidueAngleBuilder',
    'InterResidueTorsionBuilder',
    'InterResiduePlaneBuilder',
    'MONOMER_LIB_PATH'
]
