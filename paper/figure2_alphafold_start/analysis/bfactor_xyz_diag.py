#!/usr/bin/env python
"""Diagnose why torchref_norb trails phenix on the AF-start arm.

Hypothesis (user): the gap is mostly B-factors, not geometry. Phenix lets
B-factors spread; torchref's ADP restraints keep them pinned near the
(narrow, pLDDT-derived) AF starting values.

For each solved code we line up the three models by (chain, resseq, atom name):
    af_initial = placed/{code}_af.pdb        (start, B = pLDDT-derived)
    torchref   = runs/torchref_norb/results/{code}/default/refined.pdb
    phenix     = runs/phenix_norb/{code}/{code}_refined_001.pdb

and report, on the common atom set:
  - B-factor: mean / std for each model  -> does the spread open up?
  - dB std ratio torchref/phenix          -> who moved B more
  - CA XYZ RMSD torchref-vs-phenix         -> is geometry actually similar?
  - CA XYZ RMSD each-vs-af                  -> who moved coords more
"""
import csv
import sys
from pathlib import Path

import gemmi
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLACED = ROOT / "placed"
TR = ROOT / "runs/torchref_norb/results"
PX = ROOT / "runs/phenix_norb"
OUT = ROOT / "runs/metrics"


def atom_map(path):
    """{(chain, seqid, atomname): (b, x, y, z)} for the first model, no waters."""
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    m = {}
    model = st[0]
    for chain in model:
        for res in chain:
            if res.is_water():
                continue
            for at in res:
                key = (chain.name, res.seqid.num, res.seqid.icode, at.name)
                m[key] = (at.b_iso, at.pos.x, at.pos.y, at.pos.z)
    return m


def common(*maps):
    keys = set(maps[0])
    for mm in maps[1:]:
        keys &= set(mm)
    return sorted(keys)


def main(codes):
    rows = []
    for code in codes:
        af = PLACED / f"{code}_af.pdb"
        tr = TR / code / "default" / "refined.pdb"
        px = PX / code / f"{code}_refined_001.pdb"
        if not (af.exists() and tr.exists() and px.exists()):
            continue
        try:
            ma, mt, mp = atom_map(af), atom_map(tr), atom_map(px)
        except Exception as e:
            print(f"{code}: read fail {e}", file=sys.stderr)
            continue
        keys = common(ma, mt, mp)
        if len(keys) < 50:
            continue
        a = np.array([ma[k] for k in keys])  # (N,4) b,x,y,z
        t = np.array([mt[k] for k in keys])
        p = np.array([mp[k] for k in keys])
        ca = np.array([k[3] == "CA" for k in keys])

        ba, bt, bp = a[:, 0], t[:, 0], p[:, 0]
        # XYZ RMSD on the common CA set (models already in the same frame:
        # all started from the same placed model, no refit applied)
        def rmsd(u, v, mask):
            d = u[mask, 1:] - v[mask, 1:]
            return float(np.sqrt((d * d).sum(1).mean()))

        rows.append(dict(
            code=code,
            n=len(keys),
            n_ca=int(ca.sum()),
            b_af_mean=ba.mean(), b_af_std=ba.std(),
            b_tr_mean=bt.mean(), b_tr_std=bt.std(),
            b_px_mean=bp.mean(), b_px_std=bp.std(),
            # how much each refiner changed B relative to the start
            dB_tr_std=float((bt - ba).std()),
            dB_px_std=float((bp - ba).std()),
            # B agreement between refiners
            b_corr_tr_px=float(np.corrcoef(bt, bp)[0, 1]),
            # coordinate movement
            ca_rmsd_tr_px=rmsd(t, p, ca),
            ca_rmsd_tr_af=rmsd(t, a, ca),
            ca_rmsd_px_af=rmsd(p, a, ca),
        ))
        print(f"{code}: B std af={ba.std():.1f} tr={bt.std():.1f} px={bp.std():.1f} "
              f"| CA RMSD tr-px={rows[-1]['ca_rmsd_tr_px']:.2f} "
              f"tr-af={rows[-1]['ca_rmsd_tr_af']:.2f} px-af={rows[-1]['ca_rmsd_px_af']:.2f}")

    if not rows:
        print("no rows")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / "bfactor_xyz_diag.csv"
    with open(fn, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    arr = {k: np.array([r[k] for r in rows], float) for k in rows[0] if k != "code"}
    print(f"\n=== medians over n={len(rows)} structures ===")
    print(f"B std:   af={np.median(arr['b_af_std']):.1f}  "
          f"torchref={np.median(arr['b_tr_std']):.1f}  "
          f"phenix={np.median(arr['b_px_std']):.1f}")
    print(f"B std ratio torchref/phenix (median): "
          f"{np.median(arr['b_tr_std']/arr['b_px_std']):.2f}")
    print(f"dB std from start:  torchref={np.median(arr['dB_tr_std']):.1f}  "
          f"phenix={np.median(arr['dB_px_std']):.1f}")
    print(f"B corr torchref-phenix (median): {np.median(arr['b_corr_tr_px']):.2f}")
    print(f"CA RMSD torchref-phenix (median): {np.median(arr['ca_rmsd_tr_px']):.2f} A")
    print(f"CA RMSD torchref-af  (median): {np.median(arr['ca_rmsd_tr_af']):.2f} A")
    print(f"CA RMSD phenix-af    (median): {np.median(arr['ca_rmsd_px_af']):.2f} A")
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        codes = sys.argv[1:]
    else:
        codes = sorted(p.name for p in TR.iterdir() if p.is_dir())
    main(codes)
