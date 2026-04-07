#!/usr/bin/env python3
"""Strip alternate conformations from a PDB file.

For each residue with altlocs, keeps the conformer with the higher
occupancy. If occupancies are equal, keeps altloc A.  Atoms without
altloc are always kept.

Usage
-----
::

    torchref.strip-altlocs input.pdb output.pdb
"""

import argparse
import sys

import pandas as pd

from torchref.io import pdb


def strip_altlocs(df):
    """Remove alternate conformations, keeping the higher-occupancy conformer.

    Parameters
    ----------
    df : pd.DataFrame
        PDB dataframe with altloc column.

    Returns
    -------
    pd.DataFrame
        Dataframe with altlocs removed.
    """
    no_alt = df["altloc"].isna() | df["altloc"].isin(["", " "])

    if no_alt.all():
        return df.copy()

    has_alt = df[~no_alt]
    res_cols = ["chainid", "resseq", "icode", "resname"]

    # For each residue, find which altloc has the highest mean occupancy
    best_altloc = (
        has_alt.groupby(res_cols + ["altloc"])["occupancy"]
        .mean()
        .reset_index()
        .sort_values(["occupancy", "altloc"], ascending=[False, True])
        .drop_duplicates(subset=res_cols, keep="first")
        .set_index(res_cols)["altloc"]
    )

    # Keep atoms without altloc + atoms matching the best altloc
    keep_mask = no_alt.copy()
    for _, row in has_alt.iterrows():
        key = tuple(row[c] for c in res_cols)
        if key in best_altloc.index and row["altloc"] == best_altloc[key]:
            keep_mask[row.name] = True

    result = df[keep_mask].copy()
    result["altloc"] = ""
    result["serial"] = range(1, len(result) + 1)
    return result


def main():
    parser = argparse.ArgumentParser(
        prog="torchref.strip-altlocs",
        description="Strip alternate conformations from a PDB file, "
                    "keeping the conformer with the highest occupancy.",
    )
    parser.add_argument("input", help="Input PDB file")
    parser.add_argument("output", help="Output PDB file")
    args = parser.parse_args()

    df = pdb.load_as_dataframe(args.input)
    n_before = len(df)

    altlocs = df["altloc"].unique()
    altlocs = [a for a in altlocs if a != "" and pd.notna(a)]
    n_residues_with_alt = df[df["altloc"].isin(altlocs)].groupby(
        ["chainid", "resseq"]
    ).ngroups

    result = strip_altlocs(df)
    result.attrs = df.attrs
    n_after = len(result)

    print(f"Input:  {n_before} atoms, {n_residues_with_alt} residues with altlocs")
    print(f"Output: {n_after} atoms (removed {n_before - n_after})")

    pdb.write(result, args.output)
    print(f"Written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
