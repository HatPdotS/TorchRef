"""Place protein chains of a deposited structure into its own data, one at a time.

The pose panel uses the whole deposited model as the search model, which is the
easiest case there is. This takes the model apart: the search model is a single
chain's ATOM records (no HETATM, so no ligands, ions or waters), reoriented at
random, and the question is whether the pipeline puts that chain back where it
was. Three modes:

``single``
    Place one chain into the data with nothing fixed. On a homodimer the
    sibling's site is an equally correct answer -- the data cannot tell the two
    apart -- and the row says which site was taken.
``oracle``
    Place one chain with every other protein chain fixed at its **deposited**
    position. Isolates the fixed-component machinery from the first copy's own
    error, and is how a fragment too small for the rotation function is tested:
    ``--inject-true`` hands the pipeline the true orientation as its only
    rotation candidate, separating the phased translation function from the
    rotation problem.
``sequential``
    The real thing: chains by size, largest first; place, fix the **placed**
    model, place the next. A sequence-identical chain reuses the first run's
    rotation shortlist. Each chain is judged against the sites not yet filled
    (its own, or a sequence-identical sibling's); the run succeeds when every
    site is filled once.

With a fixed component the origin is pinned, so the pose metric then allows
symmetry images and lattice translations only.
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

SUCCESS_DEG = 8.0
SUCCESS_A = 4.0
MIN_RESIDUES = 30


def chain_selection(model, chain: str) -> str:
    """Phenix-style selection for the chain's ATOM records only."""
    pdb = model.pdb
    het = sorted(set(pdb.loc[(pdb["chainid"] == chain) & (pdb["ATOM"] == "HETATM"),
                             "resname"].astype(str)))
    sel = f"chain {chain}"
    if het:
        sel += " and not (" + " or ".join(f"resname {r}" for r in het) + ")"
    return sel


def protein_chains(model):
    """``[(chain_id, n_residues)]`` for chains with at least MIN_RESIDUES ATOM residues, largest first."""
    pdb = model.pdb
    atoms = pdb[pdb["ATOM"].astype(str) == "ATOM"]
    out = []
    for c in sorted(set(atoms["chainid"].astype(str))):
        n = int(atoms[atoms["chainid"].astype(str) == c][["resseq"]].drop_duplicates().shape[0])
        if n >= MIN_RESIDUES:
            out.append((c, n))
    return sorted(out, key=lambda cn: -cn[1])


def residue_sequence(model, chain: str) -> dict:
    pdb = model.pdb
    sub = pdb[(pdb["chainid"].astype(str) == chain) & (pdb["ATOM"].astype(str) == "ATOM")]
    return {int(r): str(n) for r, n in zip(sub["resseq"], sub["resname"])}


def same_sequence(model, a: str, b: str) -> bool:
    sa, sb = residue_sequence(model, a), residue_sequence(model, b)
    common = set(sa) & set(sb)
    if len(common) < 0.8 * min(len(sa), len(sb)):
        return False
    return sum(sa[r] == sb[r] for r in common) >= 0.9 * len(common)


def sibling_coordinates(model, chain_model, chain: str):
    """Deposited coordinates of every other chain with the same residue sequence,
    matched atom for atom to ``chain_model``'s atoms by (resseq, atom name).

    Returns ``[(chain_id, idx_own, xyz_other, n_matched)]`` with ``idx_own``
    indexing ``chain_model``'s atoms and ``xyz_other`` the matching atoms of the
    other chain, in that order.
    """
    pdb = model.pdb
    own = chain_model.pdb
    out = []
    for other in sorted(set(str(c) for c in pdb["chainid"])):
        if other == chain or not same_sequence(model, chain, other):
            continue
        pos = np.flatnonzero(((pdb["chainid"].astype(str) == other)
                              & (pdb["ATOM"].astype(str) == "ATOM")).values)
        sub = pdb.iloc[pos]
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


def true_orientation_peak(R_true: torch.Tensor):
    """The rotation candidate the pipeline would need to undo ``R_true``."""
    from torchref.experimental.alignment.frf.rotation_utils import (
        edmonds_euler_from_rotation_matrix)
    from torchref.experimental.alignment.frf.types import RotationPeak
    a, b, g = edmonds_euler_from_rotation_matrix(R_true.to(torch.float64))
    return RotationPeak(alpha=a, beta=b, gamma=g, score=0.0, sigma=0.0)


def first_true_index(sols, R_true, sym):
    from torchref.experimental.alignment.frf.rotation_utils import (
        rotation_angular_distance_deg)
    R_t = R_true.to(torch.float64)
    for s in sorted(sols, key=lambda s: s.candidate_index):
        R = torch.as_tensor(s.rotation, dtype=torch.float64)
        d = min(rotation_angular_distance_deg(R.T @ R_t, sym[k]) for k in range(sym.shape[0]))
        if d <= SUCCESS_DEG:
            return s.candidate_index
    return -1


def place_chain(data, chain_model, seed, *, fixed=(), candidates=None,
                n_rotation_candidates=10, verbose=0):
    """Reorient the chain at random and run the pipeline. Returns (pipe, sols, R_true, seconds)."""
    from torchref.experimental.alignment import MolecularReplacementPipeline
    canonical = chain_model.xyz().clone()
    R_true = random_rotation(seed)
    search = chain_model.copy()
    search.spacegroup = "P 1"
    search = search.copy().rotate(R_true.to(chain_model.dtype_float), center=canonical.mean(0))
    t0 = time.time()
    pipe = MolecularReplacementPipeline(
        data, search, d_min=4.0, d_max=15.0, n_shells=20, n_rotation_peaks=200,
        n_rotation_candidates=n_rotation_candidates, verbose=verbose,
        fixed=list(fixed),
    )
    sols = pipe.run(do_translation=True, candidates=candidates)
    return pipe, sols, R_true, time.time() - t0


def judge(placed_xyz, canonical, sibs, sites_open, data, *, origin_free):
    """Which open site (own or sibling) the placement lands on, with its errors."""
    best = (None, float("inf"), float("inf"))
    for site, idx_own, xyz_site in sites_open:
        if idx_own is None:
            r, t = pose_error(placed_xyz, xyz_site, data.cell, data.spacegroup,
                              allow_origin_freedom=origin_free)
        else:
            r, t = pose_error(placed_xyz[idx_own], xyz_site, data.cell, data.spacegroup,
                              allow_origin_freedom=origin_free)
        if r <= SUCCESS_DEG and t <= SUCCESS_A and t < best[2]:
            best = (site, r, t)
    if best[0] is None:
        r, t = pose_error(placed_xyz, canonical, data.cell, data.spacegroup,
                          allow_origin_freedom=origin_free)
        return None, r, t
    return best


def frame_of(placed_xyz, canonical_xyz, cell, spacegroup):
    """The frame a first placement chose: the symmetry image ``k`` and the
    fractional origin shift ``u`` with ``placed ~ S_k canonical + t_k + u``.

    The first copy is determined only up to the group's origin freedom; every
    copy after it is placed relative to that choice, so the deposited sites of
    the later chains have to be moved into the same frame before they are
    compared with the origin pinned.
    """
    P = canonical_xyz.detach().cpu().to(torch.float64)
    Q = placed_xyz.detach().cpu().to(torch.float64)
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = torch.linalg.svd(Qc.T @ Pc)
    d = torch.sign(torch.det(U @ Vt))
    Rk = U @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.float64)) @ Vt
    B = cell.fractional_matrix.detach().cpu().to(torch.float64)
    Binv = torch.linalg.inv(B)
    S = spacegroup.matrices.detach().cpu().to(torch.float64)
    T = spacegroup.translations.detach().cpu().to(torch.float64)
    ang = torch.einsum("kij,ij->k", B @ S @ Binv, Rk)
    k = int(ang.argmax())
    u = (Binv @ Q.mean(0)) - (S[k] @ (Binv @ P.mean(0)) + T[k])
    return k, u - u.round()


def into_frame(xyz_cart, k, u, cell, spacegroup):
    """Deposited coordinates moved into the frame of :func:`frame_of`."""
    X = xyz_cart.detach().cpu().to(torch.float64)
    B = cell.fractional_matrix.detach().cpu().to(torch.float64)
    Binv = torch.linalg.inv(B)
    S = spacegroup.matrices.detach().cpu().to(torch.float64)[k]
    T = spacegroup.translations.detach().cpu().to(torch.float64)[k]
    frac = (X @ Binv.T) @ S.T + T + u
    return (frac @ B.T).to(xyz_cart.dtype)


def row(mode, pdb, chain, trial, n_res, frac, sols, site, r, t, first_k, secs, extra=""):
    llg0 = sols[0].llg_score
    llg1 = sols[1].llg_score if len(sols) > 1 else float("nan")
    print(f"ROW mode={mode} pdb={pdb} chain={chain} trial={trial} n_res={n_res} "
          f"frac_atoms={frac:.2f} rot_deg={r:.2f} trans_A={t:.2f} ok={int(site is not None)} "
          f"site={site or '-'} first_true_k={first_k} llg0={llg0:.0f} llg1={llg1:.0f} "
          f"clash={sols[0].clash_fraction:.2f} seconds={secs:.1f}{extra}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True, choices=list(BENCH_PDBS))
    ap.add_argument("--chain", default=None, help="single/oracle: the chain to place")
    ap.add_argument("--mode", default="single", choices=("single", "oracle", "sequential"))
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--n-rotation-candidates", type=int, default=10)
    ap.add_argument("--inject-true", action="store_true",
                    help="oracle: hand the pipeline the true orientation as its only candidate")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    model, data = load_case(args.pdb)
    n_all = int(model.xyz().shape[0])
    sym = cartesian_symops(data.spacegroup, data.cell)
    chains = protein_chains(model)

    if args.mode in ("single", "oracle"):
        assert args.chain, "--chain is required"
        chain_model = model.select(chain_selection(model, args.chain))
        canonical = chain_model.xyz().clone()
        n_res = int(chain_model.pdb[["resseq"]].drop_duplicates().shape[0])
        frac = canonical.shape[0] / n_all
        seed = seed_for(args.pdb, args.trial) + 7919 * (ord(args.chain[0]) % 26)
        sibs = sibling_coordinates(model, chain_model, args.chain)
        fixed, candidates, extra = (), None, ""
        if args.mode == "oracle":
            others = [c for c, _ in chains if c != args.chain]
            fixed = [model.select(chain_selection(model, c)) for c in others]
            extra = f" fixed={','.join(others)}"
            if args.inject_true:
                R_true = random_rotation(seed)
                candidates = [true_orientation_peak(R_true)]
                extra += " injected=1"
        pipe, sols, R_true, secs = place_chain(
            data, chain_model, seed, fixed=fixed, candidates=candidates,
            n_rotation_candidates=args.n_rotation_candidates, verbose=args.verbose)
        placed = sols[0].model.xyz()
        if args.mode == "single":
            sites = [(args.chain, None, canonical)] + [(o, i, x) for o, i, x, _ in sibs]
            site, r, t = judge(placed, canonical, sibs, sites, data, origin_free=True)
        else:
            site, r, t = judge(placed, canonical, sibs, [(args.chain, None, canonical)],
                               data, origin_free=False)
        first_k = first_true_index(sols, R_true, sym) if candidates is None else 0
        row(args.mode, args.pdb, args.chain, args.trial, n_res, frac, sols, site, r, t,
            first_k, secs, extra)
        return 0

    # --- sequential ---------------------------------------------------------
    placed_models = []
    open_sites = {c: None for c, _ in chains}       # chain -> None (open) or placer
    shortlist, shortlist_chain = None, None
    frame = None                                    # (k, u) chosen by the first placement
    all_ok = True
    for chain, n_res in chains:
        chain_model = model.select(chain_selection(model, chain))
        canonical = chain_model.xyz().clone()
        frac = canonical.shape[0] / n_all
        seed = seed_for(args.pdb, args.trial) + 7919 * (ord(chain[0]) % 26)
        candidates = None
        if shortlist is not None and same_sequence(model, chain, shortlist_chain):
            # A reused shortlist is a list of orientations of the FIRST search
            # model, so the second copy has to start from the same frame: the
            # same random reorientation, as one model file placed twice would.
            candidates = shortlist
            seed = seed_for(args.pdb, args.trial) + 7919 * (ord(shortlist_chain[0]) % 26)
        pipe, sols, R_true, secs = place_chain(
            data, chain_model, seed, fixed=placed_models, candidates=candidates,
            n_rotation_candidates=args.n_rotation_candidates, verbose=args.verbose)
        if shortlist is None:
            shortlist, shortlist_chain = pipe.rotation_candidates, chain
        placed = sols[0].model.xyz()
        sibs = sibling_coordinates(model, chain_model, chain)
        sites = [(c, None, canonical) for c in [chain] if open_sites[c] is None]
        sites += [(o, i, x) for o, i, x, _ in sibs if o in open_sites and open_sites[o] is None]
        if frame is not None:
            # Later chains are judged in the frame the first placement chose.
            k, u = frame
            sites = [(c, i, into_frame(x, k, u, data.cell, data.spacegroup)) for c, i, x in sites]
            canonical_cmp = into_frame(canonical, k, u, data.cell, data.spacegroup)
        else:
            canonical_cmp = canonical
        site, r, t = judge(placed, canonical_cmp, sibs, sites, data,
                           origin_free=(frame is None))
        if frame is None and site is not None:
            ref = canonical if site == chain else next(x for o, i, x, _ in sibs if o == site)
            idx = None if site == chain else next(i for o, i, x, _ in sibs if o == site)
            frame = frame_of(placed if idx is None else placed[idx], ref, data.cell, data.spacegroup)
        first_k = first_true_index(sols, R_true, sym) if candidates is None else -2
        row("sequential", args.pdb, chain, args.trial, n_res, frac, sols, site, r, t,
            first_k, secs, f" n_fixed={len(placed_models)} reused={int(candidates is not None)}")
        if site is None:
            all_ok = False
        else:
            open_sites[site] = chain
        placed_models.append(sols[0].model)
    print(f"SUMMARY mode=sequential pdb={args.pdb} trial={args.trial} "
          f"chains={len(chains)} ok={int(all_ok)} "
          f"filled={','.join(f'{s}<-{p}' for s, p in open_sites.items() if p)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
