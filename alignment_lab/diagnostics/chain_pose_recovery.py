"""Place one protein chain of a deposited structure into its own data.

The pose panel uses the whole deposited model as the search model, which is the
easiest case there is. This takes the model apart: the search model is a single
chain's ATOM records (no HETATM, so no ligands, ions or waters), reoriented at
random, and the question is whether the pipeline puts that chain back where it
was. On a multi-chain structure the search model is then a fraction of the
asymmetric unit, which dilutes the rotation function's signal and breaks the
likelihood's assumption that the model accounts for all the scattering.

Success is the chain's pose against its own deposited coordinates, modulo the
crystal symmetry -- or against any other chain with the same sequence: in a
homodimer the data cannot tell the two sites apart, so a chain placed on its
sibling's site is a correct molecular-replacement solution, and the row says
which site was taken. The row also carries the likelihood margin between the
winner and the runner-up and the rotation-function index of the first correct
candidate, which is where a weaker model will show first.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import (BENCH_PDBS, cartesian_symops, load_case, pose_error,  # noqa: E402
                 random_rotation, seed_for)


def chain_selection(model, chain: str) -> str:
    """Phenix-style selection for the chain's ATOM records only."""
    pdb = model.pdb
    het = sorted(set(pdb.loc[(pdb["chainid"] == chain) & (pdb["ATOM"] == "HETATM"),
                             "resname"].astype(str)))
    sel = f"chain {chain}"
    if het:
        sel += " and not (" + " or ".join(f"resname {r}" for r in het) + ")"
    return sel


def sibling_coordinates(model, chain_model, chain: str):
    """Deposited coordinates of every other chain with the same residue sequence,
    matched atom for atom to ``chain_model``'s atoms by (resseq, atom name).

    Returns ``[(chain_id, idx_own, xyz_other, n_matched)]`` with ``idx_own``
    indexing ``chain_model``'s atoms and ``xyz_other`` the matching atoms of the
    other chain, in that order.
    """
    pdb = model.pdb
    own = chain_model.pdb
    own_res = {int(r): str(n) for r, n in zip(own["resseq"], own["resname"])}
    out = []
    for other in sorted(set(str(c) for c in pdb["chainid"])):
        if other == chain:
            continue
        pos = np.flatnonzero(((pdb["chainid"].astype(str) == other)
                              & (pdb["ATOM"].astype(str) == "ATOM")).values)
        if pos.size == 0:
            continue
        sub = pdb.iloc[pos]
        other_res = {int(r): str(n) for r, n in zip(sub["resseq"], sub["resname"])}
        common = set(own_res) & set(other_res)
        if len(common) < 0.8 * len(own_res):
            continue
        ident = sum(own_res[r] == other_res[r] for r in common)
        if ident < 0.9 * len(common):
            continue
        key = {(int(r), str(n).strip()): i for i, (r, n) in
               enumerate(zip(sub["resseq"], sub["name"]))}
        idx_own, idx_other = [], []
        for i, (r, n) in enumerate(zip(own["resseq"], own["name"])):
            j = key.get((int(r), str(n).strip()))
            if j is not None:
                idx_own.append(i); idx_other.append(j)
        if len(idx_own) < 0.5 * len(own):
            continue
        xyz_other = model.xyz()[torch.as_tensor(pos)][torch.as_tensor(idx_other)]
        out.append((other, torch.as_tensor(idx_own), xyz_other, len(idx_own)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True, choices=list(BENCH_PDBS))
    ap.add_argument("--chain", required=True)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--n-rotation-candidates", type=int, default=10)
    ap.add_argument("--success-deg", type=float, default=8.0)
    ap.add_argument("--success-A", type=float, default=4.0)
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    from torchref.experimental.alignment import MolecularReplacementPipeline
    from torchref.experimental.alignment.frf.rotation_utils import (
        rotation_angular_distance_deg)

    model, data = load_case(args.pdb)
    n_all = int(model.xyz().shape[0])
    chain_model = model.select(chain_selection(model, args.chain))
    canonical = chain_model.xyz().clone()
    n_res = int(chain_model.pdb[["chainid", "resseq"]].drop_duplicates().shape[0])
    frac = canonical.shape[0] / n_all
    seed = seed_for(args.pdb, args.trial) + 7919 * (ord(args.chain[0]) % 26)
    R_true = random_rotation(seed)
    sym = cartesian_symops(data.spacegroup, data.cell)

    search = chain_model.copy()
    search.spacegroup = "P 1"
    search = search.copy().rotate(R_true.to(model.dtype_float), center=canonical.mean(0))
    t0 = time.time()
    pipe = MolecularReplacementPipeline(
        data, search, d_min=4.0, d_max=15.0, n_shells=20, n_rotation_peaks=200,
        n_rotation_candidates=args.n_rotation_candidates, verbose=args.verbose,
    )
    sols = pipe.run(do_translation=True)
    secs = time.time() - t0
    placed = sols[0].model.xyz()
    rot, trans = pose_error(placed, canonical, data.cell, data.spacegroup)
    site = args.chain
    ok = rot <= args.success_deg and trans <= args.success_A
    if not ok:
        # A homodimer's sibling site is an equally correct answer.
        for other, idx_own, xyz_other, n_match in sibling_coordinates(model, chain_model, args.chain):
            r2, t2 = pose_error(placed[idx_own], xyz_other, data.cell, data.spacegroup)
            if r2 <= args.success_deg and t2 <= args.success_A:
                rot, trans, site, ok = r2, t2, other, True
                break

    # Rotation-function index of the first candidate at the true orientation.
    R_t = R_true.to(torch.float64)
    first_true = -1
    for s in sorted(sols, key=lambda s: s.candidate_index):
        R = torch.as_tensor(s.rotation, dtype=torch.float64)
        d = min(rotation_angular_distance_deg(R.T @ R_t, sym[k]) for k in range(sym.shape[0]))
        if d <= args.success_deg:
            first_true = s.candidate_index
            break
    llg0 = sols[0].llg_score
    llg1 = sols[1].llg_score if len(sols) > 1 else float("nan")
    print(f"ROW pdb={args.pdb} chain={args.chain} trial={args.trial} n_res={n_res} "
          f"frac_atoms={frac:.2f} n_cand={args.n_rotation_candidates} rot_deg={rot:.2f} "
          f"trans_A={trans:.2f} ok={int(ok)} site={site} first_true_k={first_true} "
          f"llg0={llg0:.0f} llg1={llg1:.0f} seconds={secs:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
