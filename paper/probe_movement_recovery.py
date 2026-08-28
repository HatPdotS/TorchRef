#!/usr/bin/env python
"""Does difference refinement recover the true displacement, or overshoot it?

Nothing measured on real data can answer this, because the true light-state structure is
never known -- the published one is itself a refinement. So the displacement is *injected*
here and the refinement is asked to find it.

Construction:

  1. take a model, call it dark;
  2. displace a contiguous stretch of it by a known vector -- that is the true light state;
  3. build the merged light intensity the two-moment physics predicts,
     ``|F_D + alpha dF|^2 + sigma_alpha^2 |dF|^2``, with a chosen alpha and lambda;
  4. add noise with sigmas grafted from a real dataset;
  5. hand the CLI the dark model as the *starting point for both states*, so the
     refinement has to discover the displacement rather than be handed it.

The recovered displacement is then compared with the injected one. A refinement that
believes the whole observed difference -- including the positive contamination that
crystal-to-crystal activation spread puts there -- should have to move the model further
than the truth to explain it.

Writes the two MTZs and prints the CLI command; run that, then re-run with ``--score`` to
compare the refined models against the injected truth.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def displaced_copy(pdb_path, out_path, chain, first, last, shift, verbose=0):
    """Write a copy of `pdb_path` with residues [first, last] of `chain` moved by `shift`.

    Returns the number of atoms actually moved, so a selection that matched nothing is
    caught rather than silently producing a zero-displacement truth.
    """
    import gemmi

    st = gemmi.read_structure(str(pdb_path))
    moved = 0
    for model in st:
        for ch in model:
            if ch.name != chain:
                continue
            for res in ch:
                if first <= res.seqid.num <= last:
                    for atom in res:
                        atom.pos = gemmi.Position(
                            atom.pos.x + shift[0],
                            atom.pos.y + shift[1],
                            atom.pos.z + shift[2],
                        )
                        moved += 1
        break
    if moved == 0:
        raise ValueError(
            f"selection chain {chain} residues {first}-{last} matched no atoms; "
            f"the injected displacement would be zero"
        )
    st.write_pdb(str(out_path))
    if verbose:
        print(f"  displaced {moved} atoms by {np.linalg.norm(shift):.3f} A")
    return moved


def simulate(dark_pdb, light_pdb, reference_mtz, out_dir, alpha, lam, d_min,
             sigma_mul, seed, device):
    """Write dark.mtz / light.mtz carrying two-moment intensities plus noise."""
    from torchref import ReflectionData
    from torchref.cli._common import load_model
    from torchref.io.datasets import FcalcDataset

    ref = ReflectionData(device=str(device), verbose=0).load_mtz(str(reference_mtz))
    ref.cut_res(highres=d_min)

    md = load_model(str(dark_pdb), max_res=d_min, device=device, verbose=0)
    ml = load_model(str(light_pdb), max_res=d_min, device=device, verbose=0)

    with torch.no_grad():
        hkl = ref.hkl
        F_D = ref.structure_factors(md, recalc=True)
        F_L = ref.structure_factors(ml, recalc=True)
        dF = F_L - F_D

        sigma_alpha_sq = alpha * (1.0 - alpha) * lam
        I_dark = F_D.abs() ** 2
        I_light = (F_D + alpha * dF).abs() ** 2 + sigma_alpha_sq * dF.abs() ** 2

        frac_contam = float(
            (sigma_alpha_sq * dF.abs() ** 2 / I_light.clamp(min=1e-12)).median()
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, intensity in (("dark", I_dark), ("light", I_light)):
        ds = FcalcDataset(
            hkl=hkl.clone(), cell=ref.cell, spacegroup=ref.spacegroup, device=device
        )
        # Phase is irrelevant to the written intensities but set_fcalc wants a complex.
        ds.set_fcalc((intensity.clamp(min=0).sqrt() + 0j).to(torch.complex64))
        noisy = ds.add_noise(sigma_mul=sigma_mul, seed=seed, verbose=False)
        # Write the intensities add_noise actually drew, negatives and all.
        data = ReflectionData(device=str(device), verbose=0).from_tensors(
            hkl=hkl.clone(),
            F=noisy.fcalc_amp.clone(),
            F_sigma=noisy.fobs_sigma.clone(),
            cell=ref.cell,
            spacegroup=ref.spacegroup,
            rfree_flags=ref.rfree_flags.clone(),
            device=str(device),
            verbose=0,
        )
        data.I = noisy.I.clone()
        data.I_sigma = noisy.I_sigma.clone()
        p = out_dir / f"{name}.mtz"
        data.write_mtz(str(p))
        paths[name] = str(p)

    meta = dict(alpha=alpha, lambda_twin=lam, sigma_alpha_sq=sigma_alpha_sq,
                d_min=d_min, sigma_mul=sigma_mul, seed=seed,
                median_contamination_fraction=frac_contam, **paths)
    (out_dir / "truth.json").write_text(json.dumps(meta, indent=2))
    return meta


def score(truth_pdb, start_pdb, refined_pdbs, chain, first, last):
    """Injected vs recovered displacement over the moved residues."""
    import gemmi

    def positions(path):
        st = gemmi.read_structure(str(path))
        st.remove_hydrogens()
        out = {}
        for ch in st[0]:
            if ch.name != chain:
                continue
            for res in ch:
                if first <= res.seqid.num <= last:
                    for atom in res:
                        out[(res.seqid.num, atom.name)] = np.array(
                            [atom.pos.x, atom.pos.y, atom.pos.z]
                        )
        return out

    truth, start = positions(truth_pdb), positions(start_pdb)
    shared = sorted(set(truth) & set(start))
    injected = np.array([np.linalg.norm(truth[k] - start[k]) for k in shared]).mean()

    print(f"\ninjected displacement over the moved residues: {injected:.3f} A "
          f"({len(shared)} atoms)")
    print(f"{'arm':14s} {'recovered':>10s} {'ratio':>8s} {'err vs truth':>13s}")
    print("-" * 50)
    rows = []
    for label, path in refined_pdbs:
        if not Path(path).exists():
            print(f"{label:14s} (missing)")
            continue
        got = positions(path)
        keys = [k for k in shared if k in got]
        rec = np.array([np.linalg.norm(got[k] - start[k]) for k in keys]).mean()
        err = np.array([np.linalg.norm(got[k] - truth[k]) for k in keys]).mean()
        rows.append((label, rec, rec / injected, err))
        print(f"{label:14s} {rec:10.3f} {rec / injected:8.2f} {err:13.3f}")
    print("-" * 50)
    print("ratio > 1 means the refinement moved further than the truth")
    return rows


def score_sweep(root, chain, first, last, start_pdb, n_boot=10000, seed=0):
    """Aggregate a seed sweep, paired seed by seed.

    Paired, not pooled: every arm refines the *same* simulated dataset within a seed, so
    the seed-to-seed spread of the noise realisation is common to all arms and cancels in
    the difference. Comparing two distributions of errors instead would drown a real
    effect in variance that is not there.

    Reports the median paired difference with a bootstrap CI, which is the shape that
    survives a skewed distribution and a handful of seeds.
    """
    import gemmi

    root = Path(root)
    seeds = sorted(d for d in root.glob("seed_*") if d.is_dir())
    if not seeds:
        print(f"no seed_* directories under {root}")
        return []

    def positions(path):
        st = gemmi.read_structure(str(path))
        st.remove_hydrogens()
        out = {}
        for ch in st[0]:
            if ch.name != chain:
                continue
            for res in ch:
                if first <= res.seqid.num <= last:
                    for atom in res:
                        out[(res.seqid.num, atom.name)] = np.array(
                            [atom.pos.x, atom.pos.y, atom.pos.z]
                        )
        return out

    start = positions(start_pdb)
    per_arm = {}
    injected = []
    for sd in seeds:
        truth_p = sd / "light_truth.pdb"
        if not truth_p.exists():
            continue
        truth = positions(truth_p)
        shared = sorted(set(truth) & set(start))
        injected.append(
            np.mean([np.linalg.norm(truth[k] - start[k]) for k in shared])
        )
        for arm_dir in sorted(sd.glob("refine_*")):
            hits = sorted(arm_dir.glob("fractions_*_light.pdb"))
            if not hits:
                continue
            got = positions(hits[0])
            keys = [k for k in shared if k in got]
            if not keys:
                continue
            err = np.mean([np.linalg.norm(got[k] - truth[k]) for k in keys])
            rec = np.mean([np.linalg.norm(got[k] - start[k]) for k in keys])
            per_arm.setdefault(arm_dir.name, {})[sd.name] = (err, rec)

    inj = float(np.mean(injected))
    complete = set.intersection(*(set(v) for v in per_arm.values())) if per_arm else set()
    complete = sorted(complete)
    print(f"\ninjected displacement {inj:.3f} A;  "
          f"{len(complete)} seeds complete in all {len(per_arm)} arms")
    if len(complete) < len(seeds):
        print(f"  ({len(seeds) - len(complete)} seed(s) dropped: not all arms finished)")

    print(f"\n{'arm':<16s} {'err (A)':>16s} {'recovered/injected':>20s}")
    print("-" * 56)
    for arm in sorted(per_arm):
        e = np.array([per_arm[arm][s][0] for s in complete])
        r = np.array([per_arm[arm][s][1] for s in complete]) / inj
        print(f"{arm:<16s} {e.mean():8.4f} +- {e.std(ddof=1):5.4f} "
              f"{r.mean():14.3f} +- {r.std(ddof=1):.3f}")

    base = "refine_coh"
    if base not in per_arm:
        return per_arm
    rng = np.random.default_rng(seed)
    print(f"\nPaired against {base}, median of per-seed differences "
          f"({n_boot} bootstrap resamples)")
    print("-" * 72)
    print(f"{'arm':<16s} {'median d(err)':>14s} {'95% CI':>22s} {'seeds better':>14s}")
    for arm in sorted(per_arm):
        if arm == base:
            continue
        d = np.array([per_arm[arm][s][0] - per_arm[base][s][0] for s in complete])
        boots = np.array([
            np.median(rng.choice(d, size=len(d), replace=True)) for _ in range(n_boot)
        ])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{arm:<16s} {np.median(d):+14.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>22s} "
              f"{f'{(d < 0).sum()}/{len(d)}':>14s}")
    print("-" * 72)
    print("negative = closer to truth than the coherent refinement")
    return per_arm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--pdb", default=str(repo / "tests/files/pdb/1DAW.pdb"))
    ap.add_argument("--reference-mtz", default=str(repo / "tests/files/mtz/1DAW.mtz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--chain", default="A")
    ap.add_argument("--first", type=int, default=40)
    ap.add_argument("--last", type=int, default=52)
    ap.add_argument("--shift", type=float, nargs=3, default=[0.35, 0.20, -0.15])
    ap.add_argument("--alpha", type=float, default=0.22)
    ap.add_argument("--lambda-twin", type=float, default=0.3)
    ap.add_argument("--dmin", type=float, default=2.05)
    ap.add_argument("--sigma-mul", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--score", action="store_true",
                    help="Compare refined models against the truth (after refining).")
    ap.add_argument("--score-sweep", action="store_true",
                    help="Aggregate a seed sweep under --out, paired seed by seed.")
    args = ap.parse_args()

    out = Path(args.out)
    truth_pdb = out / "light_truth.pdb"

    if args.score_sweep:
        score_sweep(out, args.chain, args.first, args.last, args.pdb)
        return 0

    if args.score:
        arms = []
        for d in sorted(out.glob("refine_*")):
            if not d.is_dir():
                continue
            # The output prefix encodes the fraction, which varies across arms.
            hits = sorted(d.glob("fractions_*_light.pdb"))
            arms.append((d.name, str(hits[0]) if hits else str(d / "missing.pdb")))
        score(truth_pdb, args.pdb, arms, args.chain, args.first, args.last)
        return 0

    out.mkdir(parents=True, exist_ok=True)
    print(f"Injecting a displacement into chain {args.chain} "
          f"residues {args.first}-{args.last}")
    displaced_copy(args.pdb, truth_pdb, args.chain, args.first, args.last,
                   args.shift, verbose=1)

    print("Simulating two-moment intensities...")
    meta = simulate(args.pdb, truth_pdb, args.reference_mtz, out, args.alpha,
                    args.lambda_twin, args.dmin, args.sigma_mul, args.seed,
                    torch.device(args.device))
    print(f"  alpha={meta['alpha']}  lambda={meta['lambda_twin']}  "
          f"sigma_alpha^2={meta['sigma_alpha_sq']:.4f}")
    print(f"  median contamination fraction of I: "
          f"{meta['median_contamination_fraction']:.3e}")
    print(f"\nRefine from the DARK model for both states, e.g.\n")
    print(f"  torchref.difference-refine -dm {args.pdb} -lm {args.pdb} \\\n"
          f"      -dsf {meta['dark']} -lsf {meta['light']} \\\n"
          f"      --fraction {args.alpha} --dmin {args.dmin} "
          f"-o {out}/refine_coh --device cpu\n")
    print(f"then re-run this with --score --out {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
