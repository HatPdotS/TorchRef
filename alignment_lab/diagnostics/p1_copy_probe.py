"""Does the translation search need a P1 copy of the model, or just apply_symmetry=False?

`_placement_for_candidate` copies each rotated candidate, sets its space group to
P 1, and wraps it in `DirectModelEvaluator`, because `ModelFT.forward` hardcodes
``apply_symmetry=True`` and the Crowther-Blow expansion needs the SINGLE-MOLECULE
transform ``F_p1(h R_i)`` -- it applies the symmetry itself.

But the flag exists one level down, on ``SfFFT.compute_structure_factors``. If
calling that with ``apply_symmetry=False`` on the unmodified model agrees with
the P1 copy, then per candidate the copy, the space-group assignment and whatever
they rebuild are all avoidable, and the evaluator wrapper has nothing left to do.

Reports agreement and the cost of each step, including ``Model.copy`` -- which is
on record as rebuilding a map-symmetry table per symop and being half the cost of
a rotation search.
"""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)
from lab import BENCH_PDBS, load_case, random_rotation, seed_for  # noqa: E402


def _t(fn, n=3):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return min(ts), out


def main():
    for pdb in sys.argv[1:] or ["1DAW", "2DQ6"]:
        model, data = load_case(pdb)
        rot = model.copy().rotate(
            random_rotation(seed_for(pdb, 0)).to(model.dtype_float),
            center=model.xyz().mean(0))
        rot.spacegroup = data.spacegroup.hm
        hkl = data.hkl[data.get_valid_mask()]
        hkl_i = hkl.round().to(torch.int64)

        # what the pipeline does now
        t_copy, p1 = _t(lambda: rot.copy())
        t_sg, _ = _t(lambda: setattr(p1, "spacegroup", "P 1"))
        t_p1_sf, F_p1 = _t(lambda: p1(hkl_i))

        # the candidate replacement: same model, symmetry off at the FFT
        def direct():
            sf, _ = rot.fft.compute_structure_factors(
                hkl_i, *rot.get_iso(), *rot.get_aniso(), apply_symmetry=False)
            return sf
        t_direct, F_direct = _t(direct)

        # and with symmetry ON, for contrast -- this is NOT what the TF wants
        t_sym, F_sym = _t(lambda: rot(hkl_i))

        a, b = F_p1.to(torch.complex128), F_direct.to(torch.complex128)
        num = (a - b).abs().max()
        rel = float(num / a.abs().max().clamp(min=1e-30))
        agree_sym = float((a - F_sym.to(torch.complex128)).abs().max()
                          / a.abs().max().clamp(min=1e-30))
        print(f"ROW pdb={pdb} N={hkl_i.shape[0]} "
              f"copy={1000*t_copy:.1f}ms set_sg={1000*t_sg:.1f}ms "
              f"p1_sf={1000*t_p1_sf:.1f}ms direct_sf={1000*t_direct:.1f}ms "
              f"sym_sf={1000*t_sym:.1f}ms "
              f"| max_rel_diff(p1, apply_symmetry=False)={rel:.3e} "
              f"max_rel_diff(p1, symmetry_on)={agree_sym:.3e}", flush=True)


main()
