"""
CIF/mmCIF file format reading and writing.

This module provides functions for reading and writing CIF (Crystallographic
Information File) format data, including:
- Reflection data (structure factors)
- Model data (atomic coordinates)
- Restraint dictionaries

Functions
---------
read_reflections
    Read reflection data from a CIF file.
read_model
    Read atomic coordinates from a CIF file.
read_restraints
    Read restraint dictionary from a CIF file.
list_data_blocks
    List available data blocks in a CIF file.
write_map
    Write electron density map to CCP4 format.

Classes
-------
CIFReader
    Generic CIF file reader.
ReflectionCIFReader
    Reader for structure factor CIF files.
ModelCIFReader
    Reader for mmCIF coordinate files.
RestraintCIFReader
    Reader for restraint dictionary CIF files.

Examples
--------
>>> from torchref.io import cif
>>>
>>> # Reading reflections
>>> reader = cif.read_reflections('structure-sf.cif', verbose=1)
>>> data_dict, cell, spacegroup = reader()
>>>
>>> # Reading model
>>> reader = cif.read_model('structure.cif')
>>> df, cell, spacegroup = reader()
>>>
>>> # Reading restraints
>>> reader = cif.read_restraints('ALA.cif')
>>> restraints = reader.get_all_restraints()
>>>
>>> # List data blocks
>>> blocks = cif.list_data_blocks('multi-block.cif')
"""

from typing import List, Optional

import numpy as np
import torch

# Import all CIF reader classes from the existing module
from torchref.io.cif_readers import (
    CIFReader,
    ModelCIFReader,
    ReflectionCIFReader,
    RestraintCIFReader,
)


def read_reflections(
    filepath: str, data_block: Optional[str] = None, verbose: int = 0
) -> ReflectionCIFReader:
    """
    Read reflection data from a CIF file.

    Parameters
    ----------
    filepath : str
        Path to the structure factor CIF file.
    data_block : str, optional
        Name of the data block to read. If None, uses the first block.
    verbose : int, optional
        Verbosity level. Default is 0.

    Returns
    -------
    ReflectionCIFReader
        Reader object with data loaded.

    Examples
    --------
    >>> reader = cif.read_reflections('structure-sf.cif')
    >>> data_dict, cell, spacegroup = reader()
    >>> hkl = data_dict['HKL']
    """
    return ReflectionCIFReader(filepath, verbose=verbose, data_block=data_block)


def read_model(filepath: str, verbose: int = 0) -> ModelCIFReader:
    """
    Read atomic coordinates from a CIF file.

    Parameters
    ----------
    filepath : str
        Path to the mmCIF coordinate file.
    verbose : int, optional
        Verbosity level. Default is 0.

    Returns
    -------
    ModelCIFReader
        Reader object with data loaded.

    Examples
    --------
    >>> reader = cif.read_model('structure.cif')
    >>> df, cell, spacegroup = reader()
    >>> print(f"Loaded {len(df)} atoms")
    """
    return ModelCIFReader(filepath, verbose=verbose)


def read_restraints(filepath: str) -> RestraintCIFReader:
    """
    Read restraint dictionary from a CIF file.

    Parameters
    ----------
    filepath : str
        Path to the restraint dictionary CIF file.

    Returns
    -------
    RestraintCIFReader
        Reader object with restraints loaded.

    Examples
    --------
    >>> reader = cif.read_restraints('ALA.cif')
    >>> restraints = reader.get_all_restraints()
    """
    return RestraintCIFReader(filepath)


def list_data_blocks(filepath: str) -> List[str]:
    """
    List available data blocks in a CIF file.

    Parameters
    ----------
    filepath : str
        Path to the CIF file.

    Returns
    -------
    list of str
        Names of available data blocks.

    Examples
    --------
    >>> blocks = cif.list_data_blocks('multi-dataset.cif')
    >>> print(f"Available blocks: {blocks}")
    """
    reader = CIFReader(filepath)
    return reader.available_blocks


def write_map(data, cell, filepath: str, spacegroup: str = "P1") -> int:
    """
    Write a 3D numpy array or torch tensor to a CCP4 map file.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        3D array of map data.
    cell : list, numpy.ndarray, or torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma] in A and degrees.
    filepath : str
        Output CCP4 filename.
    spacegroup : str, optional
        Space group symbol. Default is 'P1'.

    Returns
    -------
    int
        Returns 1 on success.
    """
    import gemmi

    if isinstance(data, torch.Tensor):
        np_map = data.detach().cpu().numpy().astype(np.float32)
    else:
        np_map = data.astype(np.float32)

    if isinstance(cell, torch.Tensor):
        cell = cell.detach().cpu().numpy().tolist()
    elif isinstance(cell, np.ndarray):
        cell = cell.tolist()

    map_ccp = gemmi.Ccp4Map()
    map_ccp.grid = gemmi.FloatGrid(
        np_map, gemmi.UnitCell(*cell), gemmi.SpaceGroup(spacegroup)
    )
    map_ccp.setup(0.0)
    map_ccp.update_ccp4_header()
    map_ccp.write_ccp4_map(filepath)

    return 1


# Convenience aliases
read = read_reflections  # Default read is for reflections
