import pandas as pd
from torchref.io import pdb


# Altloc labels assigned to each input PDB (up to 4)
ALTLOC_LABELS = ["A", "B", "C", "D"]

# Columns that define an atom's identity within a residue
ID_COLS = ["name", "resname", "chainid", "resseq", "icode", "element"]


def _expand_altlocs(df):
    """
    Expand existing alternate conformations in a PDB dataframe.

    If a PDB has altlocs (e.g. A/B for a residue), each altloc group is
    expanded into its own full-atom copy. Atoms without altloc are shared
    across all groups.

    Returns a dict mapping original altloc label -> DataFrame with altloc cleared.
    If no altlocs exist, returns {"": df} unchanged.
    """
    altlocs = df["altloc"].unique()
    altlocs = [a for a in altlocs if a != "" and pd.notna(a)]

    if len(altlocs) == 0:
        return {"": df.copy()}

    groups = {}
    shared = df[df["altloc"] == ""]
    for alt in sorted(altlocs):
        alt_atoms = df[df["altloc"] == alt].copy()
        alt_atoms["altloc"] = ""
        # Combine: shared atoms + this altloc's atoms
        combined = pd.concat([shared, alt_atoms], ignore_index=True)
        combined = combined.sort_values(
            ["chainid", "resseq", "icode", "name"]
        ).reset_index(drop=True)
        groups[alt] = combined

    return groups


def merge_pdbs(pdb_paths, output_path, occupancies=None, template=None):
    """
    Merge multiple PDB files into a single model with alternate conformations.

    Each input PDB gets its own altloc label (A, B, C, D). Every atom from
    every input is written as an alternate conformer. If an input PDB already
    contains alternate conformations, these are expanded first - each existing
    altloc group becomes its own conformer in the output.

    Parameters
    ----------
    pdb_paths : list of str
        Paths to input PDB files (max 4).
    output_path : str
        Path for the output merged PDB file.
    occupancies : list of float, optional
        Occupancy for each input. If None, split equally (1/N).
        Must sum to 1.0 and have same length as pdb_paths.
    template : str, optional
        PDB template file to copy header from.

    Returns
    -------
    pd.DataFrame
        The merged dataframe.
    """
    n_inputs = len(pdb_paths)
    if n_inputs > 4:
        raise ValueError(f"Maximum 4 input PDBs supported, got {n_inputs}")

    if occupancies is None:
        occupancies = [1.0 / n_inputs] * n_inputs
    if len(occupancies) != n_inputs:
        raise ValueError(
            f"Number of occupancies ({len(occupancies)}) must match "
            f"number of PDB files ({n_inputs})"
        )

    # Load all PDBs and expand any existing altlocs
    conformers = []  # list of (DataFrame, occupancy) tuples
    label_idx = 0
    attrs = None

    for path, occ in zip(pdb_paths, occupancies):
        df = pdb.load_as_dataframe(path)
        if attrs is None:
            attrs = df.attrs

        groups = _expand_altlocs(df)

        if len(groups) == 1:
            # No altlocs - this PDB becomes one conformer
            conformers.append((list(groups.values())[0], occ))
        else:
            # Has altlocs - each group becomes its own conformer,
            # split this PDB's occupancy among its altloc groups
            n_groups = len(groups)
            for group_df in groups.values():
                conformers.append((group_df, occ / n_groups))

    if len(conformers) > 4:
        raise ValueError(
            f"Total conformers after expanding altlocs is {len(conformers)}, "
            f"maximum supported is 4"
        )

    # Print summary of what was found
    print(f"Input PDBs: {n_inputs}")
    print(f"Total conformers after altloc expansion: {len(conformers)}")
    for i, (df, occ) in enumerate(conformers):
        print(f"  Conformer {ALTLOC_LABELS[i]}: {len(df)} atoms, occupancy={occ:.3f}")

    # Build merged dataframe - every atom gets an altloc
    rows = []
    for i, (df, occ) in enumerate(conformers):
        label = ALTLOC_LABELS[i]
        conf = df.copy()
        conf["altloc"] = label
        conf["occupancy"] = occ
        rows.append(conf)

    merged = pd.concat(rows, ignore_index=True)

    # Sort: group altloc conformers together per atom
    merged = merged.sort_values(
        ["chainid", "resseq", "icode", "name", "altloc"]
    ).reset_index(drop=True)
    merged["serial"] = range(1, len(merged) + 1)

    # Preserve cell/spacegroup info from first input
    merged.attrs = attrs

    pdb.write(merged, output_path, template=template)
    return merged


if __name__ == "__main__":
    dark_path = '/das/work/p17/p17490/Peter/Library/torchref/paper/figure4_difference_refinement/data/8QL2.pdb'
    light_path = '/das/work/p17/p17490/Peter/Library/torchref/paper/figure4_difference_refinement/data/torchref_0p18.pdb'
    output_path = '/das/work/p17/p17490/Peter/Library/torchref/paper/figure4_difference_refinement/data/merged_dark_light.pdb'

    merged = merge_pdbs(
        [dark_path, light_path],
        output_path,
        occupancies=[0.5, 0.5],
    )
    n_total = len(merged)
    for label in ALTLOC_LABELS[:2]:
        n = (merged["altloc"] == label).sum()
        print(f"  Altloc {label}: {n} atoms")
