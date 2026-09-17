#!/usr/bin/env python
"""Measure joint dataset scaling on work and held-out reflections.

Report amplitude agreement and propagated-variance residuals before and after
fitting centered overall and anisotropic corrections with DatasetCollection.scale.
"""

import argparse
import json
from pathlib import Path

import torch

from torchref.cli.collection_difference_refine import setup_dataset_collection


def score(collection, subset: str) -> dict:
    """Return symmetric amplitude disagreement and chi-square for one subset."""
    a, b = list(collection.values())
    mask = getattr(a, subset).mask & getattr(b, subset).mask
    with torch.no_grad():
        fa, fb = a.F[mask], b.F[mask]
        variance = a.F_sigma[mask].square() + b.F_sigma[mask].square()
        valid = torch.isfinite(variance) & (variance > 0)
        residual = fa[valid] - fb[valid]
        denominator = (fa[valid].abs() + fb[valid].abs()).sum()
        return {
            "n": int(valid.sum()),
            "R_symmetric": float(2 * residual.abs().sum() / denominator),
            "chi2": float((residual.square() / variance[valid]).mean()),
        }


def main() -> int:
    """Fit the requested reflection pair and print or save its diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent / "figure4_difference_refinement"
    parser.add_argument("--dark-sf", default=str(root / "data/8QL2-sf.cif"))
    parser.add_argument("--light-sf", default=str(root / "data/7YYZ-light.mtz"))
    parser.add_argument("--dmin", type=float, default=2.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-o", "--out")
    args = parser.parse_args()
    collection = setup_dataset_collection(
        args.dark_sf, args.light_sf, args.dmin, torch.device(args.device)
    )
    from torchref import DatasetCollection

    raw_collection = DatasetCollection(device=args.device, verbose=0)
    for name, data in collection:
        raw_collection.add_dataset(name, data.raw_data())
    collection = raw_collection
    report = {"before": {s: score(collection, s) for s in ("work", "free")}}
    collection.scale()
    report["after"] = {s: score(collection, s) for s in ("work", "free")}
    report["fit"] = collection.scaling_metrics
    result = json.dumps(report, indent=2)
    print(result)
    if args.out:
        Path(args.out).write_text(result + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
