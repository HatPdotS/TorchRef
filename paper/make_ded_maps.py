#!/usr/bin/env python
"""Turn a ``torchref.difference-refine`` results MTZ into CCP4 maps for PyMOL/Coot.

Coot opens the MTZ directly (File > Auto Open MTZ, or pick the column pair), so this
exists for PyMOL, which wants a real map. Every map is computed on ``PHIC_diff``, so
maps from different runs are directly comparable -- they share phases.

Which coefficient is which:

``mDFop-DFc``
    The difference map. ``m`` is a normalised inverse-variance weight, **not** a sigma_A
    figure of merit. Contour at +-3 sigma.
``mDFop-DFc_corr``
    The same, from the activation-decontaminated light amplitude. Present only when the
    run had ``--two-moment``.
``DDF``
    ``DF_corr - DF``: the correction itself, as a map. Featureless against resolution
    means the correction is collinear with a scale or overall-B error and should be
    distrusted; structure in it is signal.
``2mDFop-DFc``
    The 2Fo-Fc analogue, for seeing the model in its density.
"""

import argparse
import sys
from pathlib import Path

import gemmi
import numpy as np

# label -> (amplitude column, phase column). Skipped silently when absent.
MAPS = {
    "ded":        ("mDFop-DFc", "PHIC_diff"),
    "ded_corr":   ("mDFop-DFc_corr", "PHIC_diff"),
    "ded2":       ("2mDFop-DFc", "PHIC_diff"),
    "ded2_corr":  ("2mDFop-DFc_corr", "PHIC_diff"),
    "ddf":        ("DDF", "PHIC_diff"),
    "wdf":        ("WDF", "PHIC_diff"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mtz", help="fractions_*_difference_data.mtz from a refine run")
    ap.add_argument("-o", "--outdir", default=".", help="where to write the .ccp4 files")
    ap.add_argument("--prefix", default="", help="prefix for the output names")
    # 3.0 matches the library's FFT oversampling; below ~2.5 the peaks shift.
    ap.add_argument("--sample-rate", type=float, default=3.0)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    mtz = gemmi.read_mtz_file(args.mtz)
    have = {c.label for c in mtz.columns}

    print(f"{args.mtz}\n{'map':12s} {'coefficient':22s} {'rms':>10s} {'peak':>10s}")
    print("-" * 58)
    for name, (f, ph) in MAPS.items():
        if f not in have or ph not in have:
            continue
        grid = mtz.transform_f_phi_to_map(f, ph, sample_rate=args.sample_rate)
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = grid
        ccp4.update_ccp4_header()
        path = out / f"{args.prefix}{name}.ccp4"
        ccp4.write_ccp4_map(str(path))
        a = np.array(grid, copy=False)
        print(f"{name:12s} {f:22s} {a.std():10.5f} {np.abs(a).max():10.5f}")
    print(f"\nwritten to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
