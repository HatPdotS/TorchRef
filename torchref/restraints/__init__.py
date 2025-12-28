"""
Restraints module for crystallographic refinement.

This module provides classes for building and managing geometry restraints
(bonds, angles, torsions, planes, chirals, VDW contacts) from CIF dictionaries.

Classes
-------
Restraints
    Main restraints handler for crystallographic model refinement.
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

from torchref.restraints.restraints import Restraints
from torchref.restraints.restraints_new import RestraintsNew
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
]
