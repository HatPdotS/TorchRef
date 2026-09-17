"""Extract the per-atom-type table from the CCP4 energy library into a bundled CSV.

The monomer library types every atom (``_chem_comp_atom.type_energy``: NH1, OC, CH3,
...) and ``ener_lib.cif`` says what each type is: its element, whether it donates or
accepts hydrogen bonds, and its van der Waals radius with and without the hydrogens it
normally carries. TorchRef reads that table from ``torchref/data/ener_lib_atoms.csv``;
this script regenerates the CSV from the library so the two cannot drift apart
unnoticed.

Run as ``python -m torchref.scripts.extract_ener_lib [path/to/ener_lib.cif]``. Without a
path the library is fetched through the monomer-library manager. The hydrogen-bond
distance table is printed for inspection; it is the source of the contact-policy
defaults and is not bundled.
"""

import csv
import sys
from pathlib import Path

import gemmi

from torchref import PATH_TORCHREF_DATA

_ATOM_COLUMNS = (
    "type",
    "weight",
    "hb_type",
    "vdw_radius",
    "vdwh_radius",
    "ion_radius",
    "element",
    "valency",
    "sp",
)

_OUT_COLUMNS = ("type", "element", "hb_type", "vdw_radius", "vdwh_radius", "ion_radius")


def _null(value: str) -> str:
    return "" if value in (".", "?") else value


def extract(ener_lib: Path, out_csv: Path) -> int:
    """Write the ``_lib_atom`` loop of ``ener_lib`` to ``out_csv``; return the row count."""
    block = gemmi.cif.read_file(str(ener_lib))[0]
    table = block.find("_lib_atom.", list(_ATOM_COLUMNS))
    rows = []
    for row in table:
        record = dict(zip(_ATOM_COLUMNS, (str(v) for v in row)))
        vdw = _null(record["vdw_radius"])
        vdwh = _null(record["vdwh_radius"]) or vdw
        rows.append(
            {
                "type": record["type"],
                "element": record["element"],
                "hb_type": record["hb_type"],
                "vdw_radius": vdw,
                "vdwh_radius": vdwh,
                "ion_radius": _null(record["ion_radius"]),
            }
        )
    with open(out_csv, "w", newline="") as handle:
        handle.write(
            "# Per-energy-type atom properties from the CCP4 monomer library "
            "ener_lib.cif (_lib_atom loop).\n"
            "# hb_type: N neither, D donor, A acceptor, B both, "
            "H hydrogen able to hydrogen-bond.\n"
            "# vdw_radius: contact radius in Angstrom; vdwh_radius: radius to use when the "
            "atom's own hydrogens are not modelled.\n"
            "# Regenerate with python -m torchref.scripts.extract_ener_lib\n"
        )
        writer = csv.DictWriter(handle, fieldnames=list(_OUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    hbond = block.find("_lib_hbond.", ["atom_type_1", "atom_type_2", "min", "dist"])
    print(f"{len(rows)} atom types written to {out_csv}")
    print("hydrogen-bond distance table (type_1, type_2, well depth, distance):")
    for row in hbond:
        print("   ", " ".join(str(v) for v in row))
    return len(rows)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        ener_lib = Path(argv[0])
    else:
        from torchref.topology.monomer.library import MonomerLibraryManager

        ener_lib = Path(MonomerLibraryManager(verbose=0).ensure_gemmi_base()) / "ener_lib.cif"
    extract(ener_lib, Path(PATH_TORCHREF_DATA) / "ener_lib_atoms.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
