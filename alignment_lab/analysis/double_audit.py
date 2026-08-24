"""Where does the rotation search create float64 / complex128 tensors?

MPS has no float64 at all, so every double-precision tensor on the compute
device is a portability blocker. Rather than guess at the list -- twice now a
confident guess about this engine has been wrong -- intercept every torch call
and report the source line that produced each double tensor, with how many and
how large.

Deliberately reports rather than asserts. Some of these are *correct* and must
stay: the spherical-Bessel ladder needs the exponent range, the J_y
eigendecomposition and the anisotropy fit are precision-critical, and anything
already on the host costs nothing. The point is to separate those from the
per-reflection arrays that are double by inheritance.
"""

from __future__ import annotations

import argparse
import collections
import sys
import traceback
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

_DOUBLE = (torch.float64, torch.complex128)


class DoubleAudit(TorchFunctionMode):
    """Attribute every double-precision tensor to the line that made it."""

    def __init__(self, package_only: str = "torchref"):
        super().__init__()
        self.package_only = package_only
        self.sites = collections.Counter()
        self.elems = collections.Counter()
        self.devices = collections.defaultdict(set)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        try:
            tensors = []
            if isinstance(out, torch.Tensor):
                tensors = [out]
            elif isinstance(out, (tuple, list)):
                tensors = [t for t in out if isinstance(t, torch.Tensor)]
            if any(t.dtype in _DOUBLE for t in tensors):
                # Innermost frame inside the package under audit, so the report
                # names our code and not torch's internals.
                site = None
                for fr in reversed(traceback.extract_stack()[:-1]):
                    if f"/{self.package_only}/" in fr.filename:
                        short = fr.filename.split(f"/{self.package_only}/", 1)[1]
                        site = f"{self.package_only}/{short}:{fr.lineno}"
                        break
                if site is not None:
                    n = sum(t.numel() for t in tensors if t.dtype in _DOUBLE)
                    self.sites[site] += 1
                    self.elems[site] += n
                    for t in tensors:
                        if t.dtype in _DOUBLE:
                            self.devices[site].add(str(t.device))
        except Exception:                        # pragma: no cover - diagnostic
            pass
        return out

    def report(self, top: int = 30) -> None:
        print(f"{'site':62s} {'calls':>7s} {'elements':>12s}  devices")
        for site, elems in self.elems.most_common(top):
            print(f"{site:62s} {self.sites[site]:>7d} {elems:>12d}  "
                  f"{','.join(sorted(self.devices[site]))}")
        print(f"\n{len(self.elems)} distinct sites, "
              f"{sum(self.elems.values())} double elements total")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="1DAW")
    ap.add_argument("--lmax-cap", type=int, default=64)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    from lab import FRFConfig, rotated_case, run_frf, seed_for

    model, data, _ = rotated_case(args.pdb, seed_for(args.pdb, 0))
    cfg = FRFConfig(n_peaks=500, lmax_cap=args.lmax_cap)
    run_frf(model, data, cfg, capture_arf=False, verbose=0)   # warm the memos

    audit = DoubleAudit()
    with audit:
        run_frf(model, data, cfg, capture_arf=False, verbose=0)
    print(f"=== {args.pdb} cap{args.lmax_cap}: double-precision sites ===")
    audit.report(args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
