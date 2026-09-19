"""Monomer dictionaries: finding them, reading them, and patching them.

Everything that turns files on disk into the ideal geometry a template carries.
:mod:`library` resolves and caches the CCP4 Monomer Library, fetching a component on
demand; :mod:`cif` reads a dictionary into DataFrames per section; :mod:`modifications`
applies the ``chem_mod`` records that change a template when a link forms.

This is the data source. What is built from it -- the connectivity, the values over its
edges -- is the rest of :mod:`torchref.topology`. Import from the defining submodule
rather than from here.

Reference: Long, F., et al. (2017). AceDRG. Acta Cryst. D73, 112-122.
"""
