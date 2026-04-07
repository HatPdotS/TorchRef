#!/usr/bin/env python3
"""Create a multi-conformer mmCIF for PDB deposition from difference refinement.

Merges dark and light model mmCIF files into a single structure with
alternate conformations (A=dark, B=light).  Uses gemmi to read and
manipulate the structures directly, preserving all PDBx metadata
(entity IDs, label_seq_id, auth columns, etc.) needed by the PDB
deposition server.

Input models must NOT contain alternate conformations — use
``torchref.strip-altlocs`` first if needed.

Usage
-----
::

    python create_joined_model_for_deposition.py \\
        --dark dark.cif --light light.cif \\
        --occ-dark 0.82 --occ-light 0.18 \\
        -o deposited.cif
"""

import argparse
import sys

import gemmi


def validate_no_altlocs(st, label):
    """Raise if the structure contains alternate conformations."""
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    if atom.altloc != "\0":
                        raise ValueError(
                            f"{label} model contains alternate conformations "
                            f"(e.g. {chain.name} {res.name} {res.seqid} "
                            f"{atom.name} altloc={atom.altloc}). "
                            f"Use torchref.strip-altlocs to remove them first."
                        )


def merge_structures(st_dark, st_light, occ_dark, occ_light):
    """Merge two gemmi Structures into a multi-conformer model.

    For each residue, atoms from the dark model get altloc A and
    atoms from the light model get altloc B.  Occupancies are set
    accordingly.

    The dark structure is used as the base (preserving all metadata).
    """
    model_dark = st_dark[0]
    model_light = st_light[0]

    # Build lookup: (chain, seqid_str, resname) -> residue for light model
    light_lookup = {}
    for chain_l in model_light:
        for res_l in chain_l:
            key = (chain_l.name, str(res_l.seqid), res_l.name)
            light_lookup[key] = res_l

    for chain_d in model_dark:
        for res_d in chain_d:
            # Set dark atoms to altloc A
            for atom in res_d:
                atom.altloc = "A"
                atom.occ = occ_dark

            # Find matching light residue
            key = (chain_d.name, str(res_d.seqid), res_d.name)
            res_light = light_lookup.get(key)
            if res_light is None:
                continue

            # Add light atoms as altloc B
            for atom in res_light:
                new_atom = atom.clone()
                new_atom.altloc = "B"
                new_atom.occ = occ_light
                res_d.add_atom(new_atom)

    return st_dark


def main():
    parser = argparse.ArgumentParser(
        description="Create a multi-conformer mmCIF for PDB deposition "
                    "from difference refinement dark/light models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dark", required=True, help="Dark model mmCIF file")
    parser.add_argument("--light", required=True, help="Light model mmCIF file")
    parser.add_argument("--occ-dark", type=float, default=0.82,
                        help="Occupancy for dark conformer (default: 0.82)")
    parser.add_argument("--occ-light", type=float, default=0.18,
                        help="Occupancy for light conformer (default: 0.18)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output mmCIF file path")

    args = parser.parse_args()

    if abs(args.occ_dark + args.occ_light - 1.0) > 0.01:
        print(f"Warning: occupancies sum to {args.occ_dark + args.occ_light:.3f}, "
              f"not 1.0", file=sys.stderr)

    # Load structures with gemmi (preserves all PDBx metadata)
    print(f"Loading dark model:  {args.dark}")
    st_dark = gemmi.read_structure(args.dark)
    n_dark = st_dark[0].count_atom_sites()
    print(f"  {n_dark} atoms")

    print(f"Loading light model: {args.light}")
    st_light = gemmi.read_structure(args.light)
    n_light = st_light[0].count_atom_sites()
    print(f"  {n_light} atoms")

    # Validate no altlocs
    validate_no_altlocs(st_dark, "Dark")
    validate_no_altlocs(st_light, "Light")

    # Merge
    merged = merge_structures(st_dark, st_light, args.occ_dark, args.occ_light)
    n_merged = merged[0].count_atom_sites()

    print(f"\nMerged: {n_merged} atoms")
    print(f"  Occupancies: A={args.occ_dark:.2f} (dark), B={args.occ_light:.2f} (light)")

    # Add ensemble description to the structure name
    merged.name = (
        f"Mixed-state ensemble: A (dark, occ={args.occ_dark:.2f}), "
        f"B (light, occ={args.occ_light:.2f})"
    )

    # Write mmCIF — gemmi preserves all PDBx categories
    merged.make_mmcif_document().write_file(args.output)
    print(f"\nWritten to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
