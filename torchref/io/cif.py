"""
CIF/mmCIF reading and writing: reflections, coordinates, restraint
dictionaries and CCP4 maps.

Thin wrappers over the readers in :mod:`torchref.io.cif_readers`. The ``read_*``
functions return a *reader object*, not the data -- call it (or one of its
``get_*`` methods) to get the parsed content::

    data_dict, cell, spacegroup = cif.read_reflections('structure-sf.cif')()
    df, cell, spacegroup = cif.read_model('structure.cif')()
    restraints = cif.read_restraints('ALA.cif').get_all_restraints()
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
        Reader object; call it for ``(data_dict, cell, spacegroup)``.
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
        Reader object; call it for ``(df, cell, spacegroup)``.
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
        Reader object; use ``get_all_restraints()`` to extract them.
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
        Names of available data blocks, in file order.
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
        np_map, gemmi.UnitCell(*cell), gemmi.find_spacegroup_by_name(spacegroup)
    )
    map_ccp.setup(0.0)
    map_ccp.update_ccp4_header()
    map_ccp.write_ccp4_map(filepath)

    return 1


def dataframe_to_gemmi_structure(df, cell, spacegroup):
    """Convert a torchref atom DataFrame to a gemmi.Structure.

    Parameters
    ----------
    df : pandas.DataFrame
        Atom DataFrame with standard torchref columns (ATOM, serial, name,
        altloc, resname, chainid, resseq, icode, x, y, z, occupancy,
        tempfactor, element, charge, anisou_flag, u11..u23).
    cell : list or numpy.ndarray
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    spacegroup : str
        Space group name (Hermann-Mauguin notation).

    Returns
    -------
    gemmi.Structure
        The constructed gemmi Structure object.
    """
    import gemmi

    st = gemmi.Structure()
    st.name = "torchref"

    if cell is not None:
        if isinstance(cell, (list, tuple)):
            st.cell = gemmi.UnitCell(*cell)
        else:
            st.cell = gemmi.UnitCell(*cell.tolist())

    if spacegroup:
        st.spacegroup_hm = str(spacegroup)

    model = gemmi.Model("1")

    # Group by chain, then by (resseq, icode, resname) for residues
    for chain_id, chain_group in df.groupby("chainid", sort=False):
        chain = gemmi.Chain(str(chain_id) if chain_id and str(chain_id) != "nan" else "A")

        for (resseq, icode, resname), res_group in chain_group.groupby(
            ["resseq", "icode", "resname"], sort=False
        ):
            residue = gemmi.Residue()
            residue.name = str(resname).strip()
            seq_str = str(int(resseq))
            icode_str = str(icode).strip() if icode and str(icode) not in ("nan", " ") else ""
            residue.seqid = gemmi.SeqId(seq_str + icode_str)

            # Set het flag based on ATOM/HETATM
            first_atom_type = res_group.iloc[0]["ATOM"]
            if str(first_atom_type).strip() == "HETATM":
                residue.het_flag = "H"
            else:
                residue.het_flag = "A"

            for _, row in res_group.iterrows():
                atom = gemmi.Atom()
                atom.name = str(row["name"]).strip()

                elem_str = str(row["element"]).strip()
                if elem_str and elem_str != "nan":
                    atom.element = gemmi.Element(elem_str)

                atom.pos = gemmi.Position(
                    float(row["x"]), float(row["y"]), float(row["z"])
                )
                atom.occ = round(float(row["occupancy"]), 2)
                atom.b_iso = round(float(row["tempfactor"]), 2)

                altloc = str(row["altloc"]).strip()
                if altloc and altloc != "nan" and altloc != ".":
                    atom.altloc = altloc[0]

                charge = row.get("charge", 0)
                if charge and float(charge) != 0:
                    atom.charge = int(float(charge))

                # Anisotropic displacement parameters
                if row.get("anisou_flag", False):
                    u11 = float(row.get("u11", 0))
                    u22 = float(row.get("u22", 0))
                    u33 = float(row.get("u33", 0))
                    u12 = float(row.get("u12", 0))
                    u13 = float(row.get("u13", 0))
                    u23 = float(row.get("u23", 0))
                    atom.aniso = gemmi.SMat33f(u11, u22, u33, u12, u13, u23)

                residue.add_atom(atom)

            chain.add_residue(residue)

        model.add_chain(chain)

    st.add_model(model)

    # Populate PDBx label columns from structure topology
    st.setup_entities()
    st.assign_subchains()

    # Build entity sequences from residue names so label_seq_id can be assigned
    for entity in st.entities:
        if entity.entity_type == gemmi.EntityType.Polymer:
            for subchain_name in entity.subchains:
                subchain = st[0].get_subchain(subchain_name)
                entity.full_sequence = [res.name for res in subchain]
                break  # one subchain is enough

    st.assign_label_seq_id()

    return st


def _cif_value(val) -> str:
    """Render one value as a CIF token, quoting it when it needs quoting.

    The unset markers ``?`` and ``.`` are passed through bare: ``gemmi.cif.quote``
    would turn them into the quoted one-character strings ``'?'`` and ``'.'``,
    which are data rather than nulls. Everything else goes through ``quote`` --
    an unquoted value containing whitespace silently splits into extra loop
    columns when the file is read back.
    """
    import gemmi

    text = str(val)
    if text in ("?", "."):
        return text
    return gemmi.cif.quote(text)


def _add_refine_categories(doc, metadata):
    """Inject a :class:`RefinementMetadata`'s categories into ``doc``, in place.

    Returns the set of category prefixes written (e.g. ``{"_refine.",
    "_software."}``) so the caller can avoid copying the same categories in
    again from another block and either clobbering or duplicating them.
    """
    block = doc.sole_block()
    cats = metadata.render_cif_categories()
    written = set()

    for cat_name, items in cats.items():
        # List values mean a loop category rather than key-value pairs.
        is_loop = any(isinstance(v, list) for v in items.values())

        if is_loop:
            # gemmi wants a prefix ('_audit_author.') plus tag suffixes.
            tags = list(items.keys())
            prefix = tags[0].rsplit(".", 1)[0] + "."
            suffixes = [t.split(".")[-1] for t in tags]
            loop = block.init_loop(prefix, suffixes)
            written.add(prefix)
            # All list values should have same length
            n_rows = max(len(v) for v in items.values() if isinstance(v, list))
            for i in range(n_rows):
                row = []
                for tag in tags:
                    val = items[tag]
                    if isinstance(val, list):
                        cell = val[i] if i < len(val) else "?"
                    else:
                        cell = val
                    row.append(_cif_value(cell))
                loop.add_row(row)
        else:
            for key, val in items.items():
                block.set_pair(key, _cif_value(val))
                written.add(key.rsplit(".", 1)[0] + ".")

    return written


def write_model(df, filepath: str, metadata=None) -> None:
    """Write atomic coordinates to mmCIF file.

    Parameters
    ----------
    df : pandas.DataFrame
        Atom DataFrame with standard torchref columns.
    filepath : str
        Output mmCIF file path.
    metadata : RefinementMetadata, optional
        Refinement statistics, title, authors etc. When given, the file is
        rebuilt so the metadata precedes the atom records.
    """
    import gemmi

    cell = df.attrs.get("cell")
    spacegroup = df.attrs.get("spacegroup", "P 1")

    # Metadata block first, then the structure block, so metadata precedes the
    # atom records in the output file.
    if metadata is not None:
        meta_doc = gemmi.cif.Document()
        meta_block = meta_doc.add_new_block("torchref")

        if cell is not None:
            if not isinstance(cell, (list, tuple)):
                cell = cell.tolist()
            meta_block.set_pair("_cell.length_a", str(cell[0]))
            meta_block.set_pair("_cell.length_b", str(cell[1]))
            meta_block.set_pair("_cell.length_c", str(cell[2]))
            meta_block.set_pair("_cell.angle_alpha", str(cell[3]))
            meta_block.set_pair("_cell.angle_beta", str(cell[4]))
            meta_block.set_pair("_cell.angle_gamma", str(cell[5]))

        if spacegroup:
            meta_block.set_pair(
                "_symmetry.space_group_name_H-M", gemmi.cif.quote(str(spacegroup))
            )

        written = _add_refine_categories(meta_doc, metadata)
        # Categories we just wrote from metadata, plus the two written above.
        # The structure block gemmi builds from the DataFrame carries its own
        # version of some of these; copying those in would clobber a pair or
        # append a second loop for the same category.
        written |= {"_cell.", "_symmetry."}

        st = dataframe_to_gemmi_structure(df, cell, spacegroup)
        struct_doc = st.make_mmcif_document()
        struct_block = struct_doc.sole_block()

        # Copy the atom loops across into the metadata block.
        for item in struct_block:
            if item.loop is not None:
                loop = item.loop
                tags = list(loop.tags)
                suffixes = [t.split(".")[-1] for t in tags]
                prefix = tags[0].rsplit(".", 1)[0] + "."
                if prefix in written:
                    continue
                new_loop = meta_block.init_loop(prefix, suffixes)
                for row_idx in range(loop.length()):
                    row = [loop[row_idx, col] for col in range(loop.width())]
                    new_loop.add_row(row)
            elif item.pair is not None:
                tag, val = item.pair
                if tag.rsplit(".", 1)[0] + "." not in written:
                    meta_block.set_pair(tag, val)

        meta_doc.write_file(filepath)
    else:
        st = dataframe_to_gemmi_structure(df, cell, spacegroup)
        doc = st.make_mmcif_document()
        doc.write_file(filepath)


# Convenience aliases
read = read_reflections  # Default read is for reflections
