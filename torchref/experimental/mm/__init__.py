"""Molecular mechanics for TorchRef models through OpenMM.

.. warning::

   Experimental -- the API may change without notice.

:class:`OpenMMAdapter` builds an OpenMM system straight from a model's context: atoms,
residues and chains from ``model.ctx.topology``, bonds from the restraint topology,
templates from OpenMM's force-field XMLs and, for residues those do not cover, GAFF2
templates made from the monomer dictionaries. It owns the map from TorchRef rows to
OpenMM particles, the ``Context``, and the exchange of coordinates and forces.
:class:`CrystalLayout` decides which copies of the atoms the system holds -- one for an
isolated model, every symmetry copy of a unit cell, or an ensemble laid out as a
supercell -- and which molecules each copy holds.

Re-exported: :class:`OpenMMAdapter`, :class:`CrystalLayout`. The helpers in
:mod:`~torchref.experimental.mm.topology` and :mod:`~torchref.experimental.mm.ligands`
are imported from their modules. OpenMM, parmed and AmberTools are imported only when a
system is built, so importing this package needs none of them.
"""

from torchref.experimental.mm.adapter import OpenMMAdapter
from torchref.experimental.mm.layout import CrystalLayout

__all__ = ["OpenMMAdapter", "CrystalLayout"]
