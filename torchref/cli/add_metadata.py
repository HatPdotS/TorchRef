#!/usr/bin/env python3 -u

"""Standalone CLI tool to add deposition metadata to PDB or mmCIF files.

Reads a structure file, applies metadata from CLI arguments and/or a JSON file (CLI wins),
and writes PDB or mmCIF according to the output extension.

Examples
--------
::

    torchref.add-metadata -i in.pdb -o out.pdb --title "My Structure" --authors "A. Person"
    torchref.add-metadata -i in.pdb -o out.cif --metadata stats.json --title "Override"
"""

import argparse
import json
import sys
from pathlib import Path

from torchref.cli._common import configure_unbuffered_output

configure_unbuffered_output()


def main():
    """Entry point for ``torchref.add-metadata``; returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Add deposition metadata to PDB or mmCIF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add title to a PDB file
  torchref.add-metadata -i structure.pdb -o deposited.pdb --title "My Structure"

  # Convert PDB to mmCIF
  torchref.add-metadata -i structure.pdb -o structure.cif

  # Apply metadata from JSON
  torchref.add-metadata -i input.pdb -o output.pdb --metadata refinement_stats.json

  # Add authors
  torchref.add-metadata -i input.cif -o output.cif --authors "J. Smith" "A. Jones"
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Input structure file (PDB or mmCIF format)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Output file path. Format determined by extension (.pdb or .cif/.mmcif)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to JSON file with metadata (RefinementMetadata dict format)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the output file header",
    )
    parser.add_argument(
        "--authors",
        type=str,
        nargs="+",
        default=None,
        help="Author names for the output file header",
    )
    parser.add_argument(
        "--r-work",
        type=float,
        default=None,
        help="R-work value to include in header",
    )
    parser.add_argument(
        "--r-free",
        type=float,
        default=None,
        help="R-free value to include in header",
    )
    parser.add_argument(
        "--resolution-high",
        type=float,
        default=None,
        help="High-resolution limit (d_min) in Angstroms",
    )
    parser.add_argument(
        "--resolution-low",
        type=float,
        default=None,
        help="Low-resolution limit (d_max) in Angstroms",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (default: 1)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Import here to avoid slow startup for --help
    from torchref.io import pdb, cif
    from torchref.io.metadata import RefinementMetadata

    # --- Build metadata ---

    # Start with pass-through from input file
    input_suffix = input_path.suffix.lower()
    if input_suffix == ".pdb":
        metadata = RefinementMetadata.from_pdb_file(str(input_path))
    elif input_suffix in (".cif", ".mmcif"):
        metadata = RefinementMetadata.from_cif_file(str(input_path))
    else:
        metadata = RefinementMetadata()

    # Layer on JSON metadata file if provided
    if args.metadata is not None:
        json_path = Path(args.metadata)
        if not json_path.exists():
            print(f"Error: Metadata file not found: {json_path}", file=sys.stderr)
            return 1
        with open(json_path, "r") as f:
            json_data = json.load(f)
        json_meta = RefinementMetadata.from_dict(json_data)
        metadata = metadata.merge(json_meta)

    # Layer on CLI argument overrides (highest precedence)
    if args.title is not None:
        metadata.title = args.title
    if args.authors is not None:
        metadata.authors = args.authors
    if args.r_work is not None:
        metadata.r_work = args.r_work
    if args.r_free is not None:
        metadata.r_free = args.r_free
    if args.resolution_high is not None:
        metadata.resolution_high = args.resolution_high
    if args.resolution_low is not None:
        metadata.resolution_low = args.resolution_low

    # --- Read input structure ---
    if args.verbose > 0:
        print(f"Reading: {input_path}")

    if input_suffix == ".pdb":
        reader = pdb.PDBReader(verbose=0).read(str(input_path))
        df, cell_list, spacegroup = reader()
        df.attrs["cell"] = cell_list
        df.attrs["spacegroup"] = spacegroup.hm if hasattr(spacegroup, "hm") else str(spacegroup)
    elif input_suffix in (".cif", ".mmcif"):
        reader = cif.read_model(str(input_path), verbose=0)
        df, cell_list, spacegroup = reader()
        df.attrs["cell"] = cell_list
        df.attrs["spacegroup"] = spacegroup.hm if hasattr(spacegroup, "hm") else str(spacegroup)
    else:
        print(f"Error: Unsupported input format: {input_suffix}", file=sys.stderr)
        return 1

    # --- Write output ---
    output_suffix = output_path.suffix.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_suffix == ".pdb":
        pdb.write(df, str(output_path), metadata=metadata)
        if args.verbose > 0:
            print(f"Written PDB: {output_path}")
    elif output_suffix in (".cif", ".mmcif"):
        cif.write_model(df, str(output_path), metadata=metadata)
        if args.verbose > 0:
            print(f"Written mmCIF: {output_path}")
    else:
        print(f"Error: Unsupported output format: {output_suffix}", file=sys.stderr)
        return 1

    if args.verbose > 0:
        meta_dict = metadata.to_dict()
        n_fields = len(meta_dict)
        print(f"Metadata fields written: {n_fields}")
        if metadata.title:
            print(f"  Title: {metadata.title}")
        if metadata.authors:
            print(f"  Authors: {', '.join(metadata.authors)}")
        if metadata.r_work is not None:
            print(f"  R-work: {metadata.r_work:.4f}")
        if metadata.r_free is not None:
            print(f"  R-free: {metadata.r_free:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
