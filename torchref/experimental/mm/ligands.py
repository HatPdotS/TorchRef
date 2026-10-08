"""GAFF2 templates, from TorchRef's own chemistry, for residues AMBER does not cover.

For each distinct chemistry among those residues a mol2 is written straight from the
model -- its atom names, atom order, coordinates and bonds, with the net charge from the
monomer dictionary's formal charges -- antechamber assigns GAFF2 atom types and partial
charges, and parmchk2 supplies missing parameters. parmed turns the result into an OpenMM
force-field XML held in memory. Nothing else is rebuilt: the template's atoms are the
model's atoms by construction, so no map back to the model is needed.

antechamber and parmchk2 are the only programs run, once per chemistry; their output is
cached under ``PATH_TORCHREF_DATA / "gaff2_cache"``.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from torchref.experimental.mm.topology import AsuTopology

#: antechamber charge methods offered: Gasteiger (empirical, always succeeds) and
#: AM1-BCC (semi-empirical QM through sqm, slower and can fail to converge).
CHARGE_METHODS = ("gas", "bcc")

#: Dictionary bond order to mol2 bond type.
_MOL2_BOND = {
    "single": "1",
    "double": "2",
    "triple": "3",
    "aromatic": "ar",
    "deloc": "1",
}

#: antechamber's perception flag (``-j``): atom types and bond orders, the orders
#: perceived from connectivity and net charge. Typing from the dictionary's Kekulé
#: orders instead (``-j 1``) makes ANP's bridging phosphorus ``py`` rather than ``p5``.
_PERCEPTION = "4"


@dataclass(eq=False)
class LigandChemistry:
    """One residue's chemistry, as antechamber is given it.

    Parameters
    ----------
    resname : str
    names, symbols : list of str
        Atom names and element symbols, in model order.
    xyz : numpy.ndarray
        Cartesian coordinates in Å, shape ``(n, 3)``.
    bonds : list of tuple
        ``(i, j, order)`` with local atom indices and a dictionary bond order.
    net_charge : int
    """

    resname: str
    names: List[str]
    symbols: List[str]
    xyz: np.ndarray
    bonds: List[Tuple[int, int, str]]
    net_charge: int

    def key(self, charge_method: str) -> str:
        """Cache key: everything antechamber's output depends on, coordinates aside."""
        content = "|".join(
            [
                self.resname,
                ",".join(f"{n}:{s}" for n, s in zip(self.names, self.symbols)),
                ",".join(f"{i}-{j}:{o}" for i, j, o in sorted(self.bonds)),
                str(self.net_charge),
                charge_method,
                _PERCEPTION,
                _antechamber_version(),
            ]
        )
        return hashlib.sha1(content.encode()).hexdigest()

    def mol2(self) -> str:
        """The residue as a Tripos mol2, atoms in model order."""
        lines = [
            "@<TRIPOS>MOLECULE",
            self.resname,
            f"{len(self.names)} {len(self.bonds)} 1 0 0",
            "SMALL",
            "NO_CHARGES",
            "",
            "@<TRIPOS>ATOM",
        ]
        for k, (name, symbol, (x, y, z)) in enumerate(
            zip(self.names, self.symbols, self.xyz), start=1
        ):
            lines.append(
                f"{k:7d} {name:<6s} {x:10.4f} {y:10.4f} {z:10.4f} {symbol:<5s} "
                f"1 {self.resname} 0.0000"
            )
        lines.append("@<TRIPOS>BOND")
        for k, (i, j, order) in enumerate(self.bonds, start=1):
            lines.append(f"{k:6d} {i + 1:5d} {j + 1:5d} {_MOL2_BOND[order]}")
        lines.append("@<TRIPOS>SUBSTRUCTURE")
        lines.append(f"     1 {self.resname}  1 RESIDUE 0 **** **** 0 ROOT")
        return "\n".join(lines) + "\n"


def gaff2_forcefield(
    asu: "AsuTopology",
    residues: Sequence[int],
    model_xyz: np.ndarray,
    *,
    charge_method: str = "gas",
    residue_charges: Optional[Dict[str, int]] = None,
    verbose: int = 0,
) -> str:
    """OpenMM force-field XML with GAFF2 templates for the listed residues.

    Parameters
    ----------
    asu : AsuTopology
    residues : sequence of int
        Residues of ``asu`` to parameterise.
    model_xyz : numpy.ndarray
        The model's Cartesian coordinates in Å, shape ``(N, 3)``; antechamber's AM1-BCC
        charges depend on the conformation given.
    charge_method : {"gas", "bcc"}
    residue_charges : dict, optional
        Net charge per residue name, replacing the dictionary's formal charges.
    verbose : int

    Returns
    -------
    str
        One XML document holding every template and the GAFF2 parameters they use.

    Raises
    ------
    ValueError
        If a residue has no monomer dictionary, is covalently bonded to another
        residue, or ``charge_method`` is unknown.
    RuntimeError
        If antechamber or parmchk2 fails.
    """
    import parmed

    if charge_method not in CHARGE_METHODS:
        raise ValueError(f"charge_method must be one of {CHARGE_METHODS}")
    chemistries: Dict[str, Tuple[LigandChemistry, str]] = {}
    for r in residues:
        chemistry = ligand_chemistry(asu, r, model_xyz, residue_charges)
        key = chemistry.key(charge_method)
        chemistries.setdefault(key, (chemistry, key))

    files = [
        _parameterise(chemistry, key, charge_method, verbose)
        for chemistry, key in chemistries.values()
    ]
    params = parmed.amber.AmberParameterSet(
        str(_gaff2_dat()), *(str(frcmod) for _, frcmod in files)
    )
    xml_params = parmed.openmm.OpenMMParameterSet.from_parameterset(params)
    for (chemistry, key), (mol2, _) in zip(chemistries.values(), files):
        template = parmed.load_file(str(mol2))
        names = [atom.name for atom in template.atoms]
        if names != chemistry.names:
            raise RuntimeError(
                f"antechamber reordered or renamed the atoms of {chemistry.resname}"
            )
        template.name = f"{chemistry.resname}-gaff2-{key[:8]}"
        xml_params.residues[template.name] = template
    buffer = io.StringIO()
    xml_params.write(buffer, write_unused=False, improper_dihedrals_ordering="amber")
    return buffer.getvalue()


def ligand_chemistry(
    asu: "AsuTopology",
    residue: int,
    model_xyz: np.ndarray,
    residue_charges: Optional[Dict[str, int]] = None,
) -> LigandChemistry:
    """Atoms, bond orders and net charge of one residue, from model and dictionary.

    Formal charges are the dictionary's, each corrected by the hydrogens the atom
    carries beyond (+1 each) or short of (-1 each) its dictionary template; the net
    charge is their sum unless ``residue_charges`` names the residue.

    Parameters
    ----------
    asu : AsuTopology
    residue : int
        Residue of ``asu``.
    model_xyz : numpy.ndarray
        Model coordinates in Å, shape ``(N, 3)``.
    residue_charges : dict, optional

    Returns
    -------
    LigandChemistry

    Raises
    ------
    ValueError
        If the residue has no dictionary or is bonded to another residue.
    """
    resname = str(asu.residue_name[residue])
    label = asu.residue_label(residue)
    start, end = int(asu.residue_start[residue]), int(asu.residue_start[residue + 1])
    entry = asu.restraints.cif_dict.get(resname)
    if entry is None:
        raise ValueError(
            f"{label} matches no force-field template and has no monomer dictionary "
            "to parameterise it from; pass its OpenMM XML in forcefield=."
        )
    inside = (asu.bonds >= start) & (asu.bonds < end)
    if (inside[:, 0] != inside[:, 1]).any():
        raise ValueError(
            f"{label} is covalently bonded to another residue; GAFF2 templates cover "
            "free molecules only. Pass an OpenMM XML for the linked pair in forcefield=."
        )
    names = [str(n) for n in asu.name[start:end]]
    symbols = ["H" if s == "D" else str(s) for s in asu.symbol[start:end]]
    order = {}
    for _, row in entry["bonds"].iterrows():
        pair = frozenset((str(row["atom1"]), str(row["atom2"])))
        order[pair] = row.get("order", "") or "single"
    bonds = []
    for i, j in asu.bonds[inside[:, 0]] - start:
        pair = frozenset((names[i], names[j]))
        bonds.append((int(min(i, j)), int(max(i, j)), order.get(pair, "single")))

    if residue_charges and resname in residue_charges:
        net = int(residue_charges[resname])
    else:
        net = _net_formal_charge(entry, names, symbols, bonds)
    rows = asu.rows[start:end]
    return LigandChemistry(
        resname=resname,
        names=names,
        symbols=symbols,
        xyz=np.asarray(model_xyz[rows], dtype=np.float64),
        bonds=bonds,
        net_charge=net,
    )


def _net_formal_charge(entry, names, symbols, bonds) -> int:
    """Sum over heavy atoms of dictionary charge + (model H − dictionary H) bonded.

    Hydrogens are counted per parent rather than matched by name, so hydrogens named
    differently from the dictionary's still count.
    """
    atoms = entry["atoms"]
    charge = {
        str(a): (0 if c != c else int(round(c)))
        for a, c in zip(atoms["atom_id"], atoms["charge"])
    }
    dictionary_h = _hydrogens_per_parent(
        [
            (str(a), str(b))
            for a, b in zip(entry["bonds"]["atom1"], entry["bonds"]["atom2"])
        ],
        {
            str(a): str(s).strip().upper() in ("H", "D")
            for a, s in zip(atoms["atom_id"], atoms["type_symbol"])
        },
    )
    model_h = _hydrogens_per_parent(
        [(names[i], names[j]) for i, j, _ in bonds],
        {n: s == "H" for n, s in zip(names, symbols)},
    )
    heavy = [n for n, s in zip(names, symbols) if s != "H"]
    return int(
        sum(
            charge.get(n, 0) + model_h.get(n, 0) - dictionary_h.get(n, 0) for n in heavy
        )
    )


def _hydrogens_per_parent(pairs, is_hydrogen) -> Dict[str, int]:
    """Number of hydrogens bonded to each heavy atom, by atom name."""
    count: Dict[str, int] = {}
    for a, b in pairs:
        h_a, h_b = is_hydrogen.get(a, False), is_hydrogen.get(b, False)
        if h_a != h_b:
            parent = b if h_a else a
            count[parent] = count.get(parent, 0) + 1
    return count


def _parameterise(
    chemistry: LigandChemistry, key: str, charge_method: str, verbose: int
) -> Tuple[Path, Path]:
    """antechamber + parmchk2 output for one chemistry, from the cache when present."""
    from torchref import PATH_TORCHREF_DATA

    cache = Path(PATH_TORCHREF_DATA) / "gaff2_cache" / chemistry.resname
    mol2, frcmod = cache / f"{key}.mol2", cache / f"{key}.frcmod"
    if mol2.exists() and frcmod.exists():
        return mol2, frcmod
    if verbose > 0:
        print(
            f"[OpenMMAdapter] antechamber: {chemistry.resname} "
            f"(net charge {chemistry.net_charge:+d}, {charge_method})"
        )
    work = Path(tempfile.mkdtemp(prefix=f"torchref_gaff2_{chemistry.resname}_"))
    try:
        (work / "in.mol2").write_text(chemistry.mol2())
        _run(
            [
                _program("antechamber"),
                "-i",
                "in.mol2",
                "-fi",
                "mol2",
                "-o",
                "out.mol2",
                "-fo",
                "mol2",
                "-at",
                "gaff2",
                "-c",
                charge_method,
                "-nc",
                str(chemistry.net_charge),
                "-j",
                _PERCEPTION,
                "-dr",
                "no",
                "-pf",
                "yes",
            ],
            work,
            "antechamber",
            chemistry.resname,
        )
        _run(
            [
                _program("parmchk2"),
                "-i",
                "out.mol2",
                "-f",
                "mol2",
                "-o",
                "out.frcmod",
                "-s",
                "gaff2",
            ],
            work,
            "parmchk2",
            chemistry.resname,
        )
        cache.mkdir(parents=True, exist_ok=True)
        # Write-then-rename keeps a concurrent build from reading a half-written file.
        for source, target in (
            (work / "out.mol2", mol2),
            (work / "out.frcmod", frcmod),
        ):
            partial = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
            shutil.copy2(source, partial)
            partial.replace(target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return mol2, frcmod


def _run(command: List[str], cwd: Path, program: str, resname: str) -> None:
    result = subprocess.run(
        command, cwd=str(cwd), capture_output=True, text=True, timeout=600
    )
    outputs = {"antechamber": "out.mol2", "parmchk2": "out.frcmod"}
    if result.returncode != 0 or not (cwd / outputs[program]).exists():
        raise RuntimeError(
            f"{program} failed for {resname}:\n{result.stdout[-2000:]}\n"
            f"{result.stderr[-1000:]}"
        )


@lru_cache(maxsize=None)
def _program(name: str) -> str:
    """Path to an AmberTools program on ``PATH`` or under ``$AMBERHOME/bin``."""
    path = shutil.which(name)
    if path:
        return path
    home = os.environ.get("AMBERHOME")
    if home and os.access(Path(home) / "bin" / name, os.X_OK):
        return str(Path(home) / "bin" / name)
    raise FileNotFoundError(
        f"{name} not found on PATH or in $AMBERHOME/bin; install AmberTools "
        "(conda install -c conda-forge ambertools) or pass the residue's OpenMM XML "
        "in forcefield=."
    )


@lru_cache(maxsize=None)
def _antechamber_version() -> str:
    """antechamber's banner line, which names its version."""
    result = subprocess.run(
        [_program("antechamber"), "-h"], capture_output=True, text=True, timeout=60
    )
    for line in (result.stdout + result.stderr).splitlines():
        if "antechamber" in line.lower() and any(ch.isdigit() for ch in line):
            return line.strip()
    return "unknown"


def _gaff2_dat() -> Path:
    """GAFF2 parameter file shipped with AmberTools."""
    candidates = []
    home = os.environ.get("AMBERHOME")
    if home:
        candidates.append(Path(home))
    candidates.append(Path(_program("antechamber")).resolve().parents[1])
    for root in candidates:
        path = root / "dat" / "leap" / "parm" / "gaff2.dat"
        if path.exists():
            return path
    raise FileNotFoundError(
        "gaff2.dat not found under $AMBERHOME or the AmberTools prefix"
    )
