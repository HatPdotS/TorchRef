#!/usr/bin/env python3
"""Prepare Phaser search model(s) for one structure's AlphaFold-start arm.

For a given PDB code, for each unique UniProt in its manifest record:
  - load the downloaded AlphaFold model (afdb/AF-{acc}.pdb),
  - align the AF sequence to the crystallized construct sequence (from the
    manifest) with gemmi,
  - keep AF residues that fall within the construct span, drop AF-only flanks /
    insertions, and accept construct residues the AF model lacks,
  - at positions where AF disagrees with the construct, reduce the residue to
    alanine (rename ALA + strip side chain to N,CA,C,O,CB), EXCEPT when either
    side is proline or the AF residue is glycine (kept as-is),
  - write search_models/{code}/{acc}_search.pdb.

Also writes search_models/{code}/components.json listing, per component, the
search-model path and the number of copies in the asymmetric unit (= number of
deposited chains sharing that UniProt) for Phaser's composition.

This runs in the torchref env (gemmi); it does NOT need phenix.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import gemmi

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
AFDB = HERE / "afdb"
SEARCH = HERE / "search_models"
DATA = HERE.parent / "data"  # paper/data/{code}/{code}.mtz

BACKBONE_CB = {"N", "CA", "C", "O", "CB"}
_CIGAR = re.compile(r"(\d+)([MID])")


def _load_manifest():
    return {r["code"]: r for r in json.loads(MANIFEST.read_text())}


def _construct_seqs_by_uniprot(record):
    """Return OrderedDict {uniprot: (construct_seq, n_copies)} for protein chains.

    n_copies = number of deposited chains sharing the UniProt. When several
    chains map to one UniProt, the longest construct sequence is used.
    """
    by_uni = OrderedDict()
    for ch in record["chains"]:
        uni = ch.get("uniprot")
        if not uni:
            continue
        seq = ch["seq"]
        if uni not in by_uni:
            by_uni[uni] = [seq, 1]
        else:
            by_uni[uni][1] += 1
            if len(seq.replace("?", "")) > len(by_uni[uni][0].replace("?", "")):
                by_uni[uni][0] = seq
    return by_uni


def _residue_flags(af_seq, construct_seq):
    """Walk the AF↔construct alignment; return one (keep, alanize) per AF residue.

    query = AF, target = construct. CIGAR: M = aligned column, I = AF-only
    (drop), D = construct-only (AF lacks it; nothing to emit).
    """
    res = gemmi.align_string_sequences(list(af_seq), list(construct_seq), [])
    cigar = res.cigar_str()

    flags = []
    qi = ti = 0  # AF index, construct index
    for count, op in _CIGAR.findall(cigar):
        count = int(count)
        if op == "M":
            for _ in range(count):
                af_aa = af_seq[qi]
                con_aa = construct_seq[ti]
                alanize = (
                    af_aa != con_aa
                    and con_aa not in "?X"          # unknown/gap in construct
                    and af_aa not in ("P", "G")     # keep Pro/Gly as-is
                    and con_aa != "P"               # don't replace a proline target
                )
                flags.append((True, alanize))
                qi += 1
                ti += 1
        elif op == "I":  # AF-only residue → drop
            for _ in range(count):
                flags.append((False, False))
                qi += 1
        elif op == "D":  # construct-only residue → AF has none
            ti += count
    return flags


def _prepare_component(acc, construct_seq, out_pdb):
    """Trim+alanize AF-{acc}.pdb against construct_seq; write out_pdb. Returns stats."""
    af_path = AFDB / f"AF-{acc}.pdb"
    st = gemmi.read_structure(str(af_path))
    chain = st[0][0]
    poly = chain.get_polymer()
    af_seq = poly.make_one_letter_sequence()

    # Drop residue-numbering-gap padding ('?') so the construct sequence is the
    # ordered resolved residues; the search model then spans the same residues
    # as the deposited chain rather than AF's prediction of disordered loops.
    construct_seq = construct_seq.replace("?", "")
    flags = _residue_flags(af_seq, construct_seq)

    out = gemmi.Structure()
    out.cell = st.cell
    out.spacegroup_hm = st.spacegroup_hm
    model = gemmi.Model("1")
    new_chain = gemmi.Chain("A")
    n_kept = n_alanized = 0
    for res, (keep, alanize) in zip(poly, flags):
        if not keep:
            continue
        if alanize:
            res.name = "ALA"
            res.trim_to_alanine()
            n_alanized += 1
        new_chain.add_residue(res)
        n_kept += 1
    model.add_chain(new_chain)
    out.add_model(model)
    out.setup_entities()
    out.write_pdb(str(out_pdb))
    return {"acc": acc, "af_len": len(af_seq), "n_kept": n_kept, "n_alanized": n_alanized}


def _mtz_labels(code):
    """Detect (F, SIGF) column labels for the structure's MTZ (prefers FP/SIGFP)."""
    mtz = gemmi.read_mtz_file(str(DATA / code / f"{code}.mtz"))
    f_cols = [c.label for c in mtz.columns if c.type == "F"]
    q_cols = [c.label for c in mtz.columns if c.type == "Q"]
    f = "FP" if "FP" in f_cols else (f_cols[0] if f_cols else None)
    sig = "SIGFP" if "SIGFP" in q_cols else (q_cols[0] if q_cols else None)
    return f, sig


def _write_phaser_keywords(code, out_dir, components, jobs=4):
    """Write a Phaser MR_AUTO keyword file referencing the *processed* models.

    The ``_processed`` models are produced by phenix.process_predicted_model in
    the SLURM task before Phaser runs (pLDDT→B conversion).
    """
    mtz = (DATA / code / f"{code}.mtz").resolve()
    f, sig = _mtz_labels(code)
    lines = [
        f"TITLE {code} AlphaFold-start MR",
        "MODE MR_AUTO",
        f"HKLIN {mtz}",
        f"LABIN F={f} SIGF={sig}",
    ]
    for comp in components:
        acc = comp["acc"]
        proc = Path(comp["search_pdb"]).with_name(f"{acc}_search_processed.pdb")
        comp["processed_pdb"] = str(proc)
        lines.append(
            f"ENSEMBLE e_{acc} PDBFILE {proc} IDENTITY 0.9"
        )
        lines.append(
            f"COMPOSITION PROTEIN SEQUENCE {comp['fasta']} NUMBER {comp['copies']}"
        )
    for comp in components:
        lines.append(f"SEARCH ENSEMBLE e_{comp['acc']} NUMBER {comp['copies']}")
    lines.append(f"ROOT {code}_phaser")
    lines.append(f"JOBS {jobs}")
    (out_dir / "phaser.keywords").write_text("\n".join(lines) + "\n")
    return f, sig


def prepare(code, manifest):
    record = manifest.get(code)
    if record is None:
        raise SystemExit(f"{code}: not in manifest")
    if not record.get("af_complete"):
        raise SystemExit(f"{code}: not af_complete; nothing to prepare")

    out_dir = SEARCH / code
    out_dir.mkdir(parents=True, exist_ok=True)

    components = []
    stats = []
    for uni, (con_seq, n_copies) in _construct_seqs_by_uniprot(record).items():
        out_pdb = out_dir / f"{uni}_search.pdb"
        s = _prepare_component(uni, con_seq, out_pdb)
        s["copies"] = n_copies
        s["construct_len"] = len(con_seq.replace("?", ""))
        stats.append(s)

        # FASTA (gap-stripped construct sequence) for Phaser COMPOSITION.
        fasta = out_dir / f"{uni}.fasta"
        fasta.write_text(f">{code}_{uni}\n{con_seq.replace('?', '')}\n")

        components.append(
            {
                "acc": uni,
                "search_pdb": str(out_pdb),
                "fasta": str(fasta),
                "copies": n_copies,
            }
        )

    f, sig = _write_phaser_keywords(code, out_dir, components)
    config = {
        "code": code,
        "mtz": str((DATA / code / f"{code}.mtz").resolve()),
        "labin_F": f,
        "labin_SIGF": sig,
        "components": components,
    }
    (out_dir / "components.json").write_text(json.dumps(components, indent=2))
    (out_dir / "mr_config.json").write_text(json.dumps(config, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("codes", nargs="+", help="PDB code(s) to prepare.")
    args = ap.parse_args()

    manifest = _load_manifest()
    for code in args.codes:
        stats = prepare(code, manifest)
        for s in stats:
            print(
                f"{code} {s['acc']}: af_len={s['af_len']} construct={s['construct_len']} "
                f"kept={s['n_kept']} alanized={s['n_alanized']} copies={s['copies']}"
            )


if __name__ == "__main__":
    sys.exit(main())
