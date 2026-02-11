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

import os
import subprocess
from pathlib import Path

from torchref import ROOT_TORCHREF
from torchref.restraints.builders import (
    AngleRestraintBuilder,
    BondRestraintBuilder,
    ChiralRestraintBuilder,
    InterResidueAngleBuilder,
    InterResidueBondBuilder,
    InterResiduePlaneBuilder,
    InterResidueTorsionBuilder,
    PlaneRestraintBuilder,
    ResidueIterator,
    RestraintBuilder,
    TorsionRestraintBuilder,
)
from torchref.restraints.restraints import RestraintsNew as Restraints

MONOMER_LIB_URL = "https://github.com/MonomerLibrary/monomers.git"
# Pin to specific commit to avoid breaking changes (e.g., SS link deletion in f1ef252bb)
MONOMER_LIB_COMMIT = "f113ca1aa"


def _ensure_monomer_library() -> Path:
    """
    Ensure the external monomer library is available, downloading if necessary.

    Downloads the monomer library and checks out a specific pinned commit to
    ensure compatibility with TorchRef's restraints system.

    Returns
    -------
    Path
        Path to the monomer library directory.

    Raises
    ------
    RuntimeError
        If the library cannot be downloaded.
    """
    lib_path = ROOT_TORCHREF / "external_monomer_library"

    if lib_path.exists():
        return lib_path

    print(f"Monomer library not found at {lib_path}")
    print("Downloading monomer library (one-time setup)...")

    try:
        # Clone the repository
        subprocess.run(
            ["git", "clone", MONOMER_LIB_URL, str(lib_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        # Checkout the pinned commit
        subprocess.run(
            ["git", "-C", str(lib_path), "checkout", MONOMER_LIB_COMMIT],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Successfully downloaded monomer library to {lib_path}")
        print(f"  Pinned to commit: {MONOMER_LIB_COMMIT}")
        return lib_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to download monomer library from {MONOMER_LIB_URL}.\n"
            f"Error: {e.stderr}\n"
            f"Please install git or manually clone the repository:\n"
            f"  git clone {MONOMER_LIB_URL} {lib_path}\n"
            f"  git -C {lib_path} checkout {MONOMER_LIB_COMMIT}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "git is not installed. Please install git and try again, or manually clone:\n"
            f"  git clone {MONOMER_LIB_URL} {lib_path}\n"
            f"  git -C {lib_path} checkout {MONOMER_LIB_COMMIT}"
        )


MONOMER_LIB_PATH = _ensure_monomer_library()


__all__ = [
    "Restraints",
    "RestraintsNew",
    "ResidueIterator",
    "RestraintBuilder",
    "BondRestraintBuilder",
    "AngleRestraintBuilder",
    "TorsionRestraintBuilder",
    "PlaneRestraintBuilder",
    "ChiralRestraintBuilder",
    "InterResidueBondBuilder",
    "InterResidueAngleBuilder",
    "InterResidueTorsionBuilder",
    "InterResiduePlaneBuilder",
    "MONOMER_LIB_PATH",
]
