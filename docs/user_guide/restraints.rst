Geometry Restraints
===================

Geometry restraints maintain chemically reasonable geometry during refinement.


Overview
--------

The :class:`~torchref.restraints.restraints.Restraints` class handles:

- Bond length restraints
- Bond angle restraints  
- Torsion angle restraints
- Planarity restraints

Restraint Setup
---------------

Restraints are automatically generated from the monomer library:

.. code-block:: python

   from torchref.restraints import Restraints

   # Automatic setup from model
   restraints = Restraints(model, cif_file=None, verbose=1)

   # Or with custom CIF file for ligands
   restraints = Restraints(model, cif_file="ligand.cif", verbose=1)

Restraint Storage
-----------------

Restraints are stored in a nested dictionary structure.
We store the indices needed to access the atoms involved, along with ideal values and sigma values.
   
.. code-block:: python

   restraints.restraints = {
       'bond': {
           'intra': {
               'indices': Tensor of shape (N_bonds, 2),
               'ideal_values': Tensor of shape (N_bonds,),
               'weights': Tensor of shape (N_bonds,),
           },
           'inter': { ... },
       },
       'angle': {
           'intra': { ... },
           'inter': { ... },
       },
       'torsion': { ... },
       'planarity': { ... },
   }

Restraint Types
---------------

Bond Restraints
^^^^^^^^^^^^^^^

Penalize deviations from ideal bond lengths:

.. math::

   E_{bond} = \sum_i w_i \left( d_i - d_i^{ideal} \right)^2

where :math:`d_i` is the observed distance and :math:`d_i^{ideal}` is the target.

Angle Restraints
^^^^^^^^^^^^^^^^

Penalize deviations from ideal bond angles:

.. math::

   E_{angle} = \sum_i w_i \left( \theta_i - \theta_i^{ideal} \right)^2

Torsion Restraints
^^^^^^^^^^^^^^^^^^

Periodic restraints for dihedral angles.

Planarity Restraints
^^^^^^^^^^^^^^^^^^^^

Restrain groups of atoms to lie in a plane (aromatic rings, peptide bonds).

Accessing Restraint Statistics
------------------------------

.. code-block:: python

   # RMSD values
   bond_rmsd = restraints.get_bond_rmsd()
   angle_rmsd = restraints.get_angle_rmsd()

   # Number of restraints (stored in the nested restraints dict)
   n_bonds = restraints.restraints['bond']['intra']['indices'].shape[0]
   n_angles = restraints.restraints['angle']['intra']['indices'].shape[0]

