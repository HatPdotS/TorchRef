#!/usr/bin/env python3
"""Strip alternate conformations from a PDB file.

Writes :meth:`torchref.model.model.Model.strip_altlocs` of the input: in each residue
(chain, resseq, icode) with several altlocs, the conformer with the highest mean
occupancy is kept (the first in sorted order on a tie), so alternates carrying
different residue names compete too. Atoms without altloc are always kept.
Occupancies are written as a ``Model`` reads them: one value per conformer, the
conformers of a residue summing to 1.

Usage
-----
::

    torchref.strip-altlocs input.pdb output.pdb
"""

import argparse
import sys

from torchref import Model


def main():
    """Entry point for ``torchref.strip-altlocs``; returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="torchref.strip-altlocs",
        description="Strip alternate conformations from a PDB file, "
                    "keeping the conformer with the highest occupancy.",
    )
    parser.add_argument("input", help="Input PDB file")
    parser.add_argument("output", help="Output PDB file")
    args = parser.parse_args()

    # Pure bookkeeping on the atom table: no reason to start an accelerator.
    model = Model(device="cpu", verbose=0).load_pdb(args.input)
    result = model.strip_altlocs()
    n_before, n_after = model.n_atoms, result.n_atoms
    n_residues_with_alt = len(model.ctx.altloc_residues())

    print(f"Input:  {n_before} atoms, {n_residues_with_alt} residues with altlocs")
    print(f"Output: {n_after} atoms (removed {n_before - n_after})")

    result.write_pdb(args.output)
    print(f"Written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
