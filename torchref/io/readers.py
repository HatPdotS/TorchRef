"""
Top-level file readers for object creation.

These thin convenience functions construct the appropriate domain object
from a file path, wrapping the existing loaders. ``read_cif`` inspects the
CIF content to dispatch between reflection data, an atomic model, an IHM
ensemble, or a restraint dictionary.

Examples
--------
::

    from torchref import read_mtz, read_cif, read_pdb

    data  = read_mtz("reflections.mtz")   # -> ReflectionData
    model = read_pdb("structure.pdb")     # -> ModelFT
    obj   = read_cif("anything.cif")      # dispatched by content
"""

from pathlib import Path
from typing import Any, Union

import gemmi

__all__ = ["read_mtz", "read_cif", "read_pdb"]


def _detect_cif_type(filepath: str) -> str:
    """
    Detect the kind of data a CIF file holds by probing its loops.

    Returns one of ``"reflections"``, ``"ihm_ensemble"``, ``"structure"``,
    or ``"restraints"``. Raises ``ValueError`` if nothing recognizable is
    found.
    """
    try:
        doc = gemmi.cif.read_file(str(filepath))
    except Exception as e:  # pragma: no cover - malformed file
        raise ValueError(f"Failed to parse CIF file: {filepath}\n  {e}") from e

    has_refln = has_atom_site = has_restraints = has_ihm = False
    for block in doc:
        if (
            block.find_loop("_refln.index_h")
            or block.find_loop("_refln_index_h")
            or block.find_value("_refln.index_h")
        ):
            has_refln = True
        if (
            block.find_loop("_atom_site.group_PDB")
            or block.find_loop("_atom_site_group_PDB")
            or block.find_value("_atom_site.group_PDB")
        ):
            has_atom_site = True
        if (
            block.find_loop("_chem_comp_atom.atom_id")
            or block.find_loop("_chem_comp_bond.atom_id_1")
            or block.find_value("_chem_comp.id")
        ):
            has_restraints = True
        if block.find_loop("_ihm_model_list.model_id") or block.find(
            ["_ihm_multi_state_modeling.state_id"]
        ):
            has_ihm = True

    # IHM ensembles also carry _atom_site, so check them first.
    if has_ihm and has_atom_site:
        return "ihm_ensemble"
    if has_refln:
        return "reflections"
    if has_atom_site:
        return "structure"
    if has_restraints:
        return "restraints"
    raise ValueError(
        f"CIF file does not contain recognizable data: {filepath}\n"
        "Expected reflection data (_refln), structure data (_atom_site), "
        "or restraint data (_chem_comp)."
    )


def read_mtz(filepath: Union[str, Path], verbose: int = 1, **load_kwargs):
    """
    Load an MTZ reflection file into a :class:`ReflectionData`.

    Parameters
    ----------
    filepath : str or Path
        Path to the MTZ file.
    verbose : int, optional
        Verbosity level. Default is 1.
    **load_kwargs
        Forwarded to :meth:`ReflectionData.load_mtz`.

    Returns
    -------
    ReflectionData
    """
    from torchref.io.datasets import ReflectionData

    data = ReflectionData(verbose=verbose)
    data.load_mtz(str(filepath), **load_kwargs)
    return data


def read_pdb(filepath: Union[str, Path], model_class=None, **model_kwargs):
    """
    Load a PDB file into a model.

    Parameters
    ----------
    filepath : str or Path
        Path to the PDB file.
    model_class : type, optional
        Model class to construct. Defaults to :class:`ModelFT`. Pass
        :class:`Model` for the plain base model.
    **model_kwargs
        Forwarded to the model constructor (e.g. ``max_res``,
        ``radius_angstrom``).

    Returns
    -------
    ModelFT (or the requested ``model_class``)
    """
    if model_class is None:
        from torchref.model import ModelFT as model_class
    model = model_class(**model_kwargs)
    model.load_pdb(str(filepath))
    return model


def read_cif(filepath: Union[str, Path], model_class=None, verbose: int = 1, **kwargs) -> Any:
    """
    Load a CIF file, dispatching on its content.

    The CIF is probed to decide what it contains:

    - reflection data (``_refln``)        -> :class:`ReflectionData`
    - IHM ensemble (``_ihm_model_list``)  -> ``(ModelCollection, IHMEnsembleMapping)``
    - atomic model (``_atom_site``)       -> ``model_class`` (default :class:`ModelFT`)
    - restraint dictionary (``_chem_comp``) -> :class:`RestraintCIFReader`

    Parameters
    ----------
    filepath : str or Path
        Path to the CIF/mmCIF file.
    model_class : type, optional
        Model class for the structure branch. Defaults to :class:`ModelFT`.
    verbose : int, optional
        Verbosity level. Default is 1.
    **kwargs
        Forwarded to the underlying loader/constructor.

    Returns
    -------
    object
        Type depends on the CIF content (see above).
    """
    path = str(filepath)
    kind = _detect_cif_type(path)

    if kind == "reflections":
        from torchref.io.datasets import ReflectionData

        data = ReflectionData(verbose=verbose)
        data.load_cif(path, **kwargs)
        return data

    if kind == "ihm_ensemble":
        from torchref.model.model_collection import ModelCollection

        return ModelCollection.from_ihm(path, verbose=verbose, **kwargs)

    if kind == "structure":
        if model_class is None:
            from torchref.model import ModelFT as model_class
        model = model_class(**kwargs)
        model.load_cif(path)
        return model

    if kind == "restraints":
        from torchref.io.cif import RestraintCIFReader

        return RestraintCIFReader(path, verbose=verbose)

    raise ValueError(f"Unrecognized CIF content in {path!r}")  # pragma: no cover
