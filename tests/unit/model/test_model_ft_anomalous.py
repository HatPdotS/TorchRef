"""Anomalous scattering in ``ModelFT``'s F_calc, pinned against independent references.

f' and f'' enter F_calc as zero-width terms of each atom's form factor, so the one
density splat and FFT give them the atom's own temperature factor (isotropic or
anisotropic) and the space-group symmetry, as for f0. The references share none of
that path:

* f' -- gemmi's ``StructureFactorCalculatorX`` with ``addends``, which adds a constant
  to an element's form factor inside gemmi's own symmetry and Debye-Waller sums.
* f'' -- gemmi has no imaginary addend, so :func:`_constant_term_sum` sums it
  explicitly in float64 over every symmetry mate with each atom's temperature factor.
  That helper is itself checked against gemmi's addends, with the real f'.

Both sides are driven by the model's own atoms (the gemmi structure is built from the
model's tensors), so reader differences cannot pose as structure-factor error. The
overall accuracy is gated against the FFT path's own error without anomalous terms,
measured against the same reference: grid sampling and truncation set the precision
this path can deliver, and the anomalous terms must not add to it. Each structure is
evaluated at its deposited resolution, on the grid production would build for it.

Structures: 3E98 (P 1 21 1, isotropic, Se), 5BOV (P 1, anisotropic, Se) and 6G9X
(P 21 21 2, anisotropic, Hg, whose f'' of 10 e makes large Bijvoet differences).
"""

from types import SimpleNamespace

import gemmi
import numpy as np
import pytest
import torch

from torchref.base.scattering.anomalous_table import get_significant_elements
from torchref.model import ModelFT

N_REFL = 400
WAVELENGTH = 1.0

#: Factor by which the anomalous terms may grow the FFT path's own error, overall and
#: in the top resolution quarter. Measured at most 1.011; terms summed over the
#: asymmetric unit without temperature factors grow it 1.9x-12x (top quarter 16x-46x).
ERROR_GROWTH = 1.05

#: Relative error allowed on the anomalous contribution itself (``F - F(f0)`` against
#: the reference's) and on the Bijvoet differences. Measured 0.03%-0.09% and
#: 0.23%-0.65%; without symmetry mates and temperature factors both exceed 100%.
TERM_TOL = 0.05

STRUCTURES = [
    pytest.param(("3E98", 2.5), id="3E98-P1211-iso"),
    pytest.param(("5BOV", 1.6), id="5BOV-P1-aniso"),
    pytest.param(("6G9X", 2.3), id="6G9X-P21212-aniso-Hg"),
]


def _model(pdb_dir, code, d_min, **kwargs) -> ModelFT:
    path = pdb_dir / f"{code}.pdb"
    if not path.exists():
        pytest.skip(f"{code}.pdb fixture not present")
    kwargs.setdefault("wavelength", WAVELENGTH)
    return ModelFT(max_res=d_min, verbose=0, device="cpu", **kwargs).load_pdb(str(path))


def _gemmi_structure(model: ModelFT) -> gemmi.Structure:
    """A gemmi structure holding exactly the model's atoms, ADPs and occupancies."""
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(*[float(v) for v in model.cell.data.tolist()])
    st.spacegroup_hm = model.spacegroup.hm
    xyz = model.xyz().detach().double().numpy()
    adp = model.adp().detach().double().numpy()
    occ = model.occupancy().detach().double().numpy()
    u = model.u().detach().double().numpy()
    aniso = model.aniso_flag.numpy()
    chain = gemmi.Chain("A")
    for i, element in enumerate(model.ctx.topology.atoms.element.tolist()):
        atom = gemmi.Atom()
        atom.name = "X"
        atom.element = gemmi.Element(element)
        atom.pos = gemmi.Position(*xyz[i])
        atom.occ = float(occ[i])
        atom.b_iso = float(adp[i])
        if aniso[i]:
            atom.aniso = gemmi.SMat33f(*[float(v) for v in u[i]])
        residue = gemmi.Residue()
        residue.name = "UNK"
        residue.seqid = gemmi.SeqId(i + 1, " ")
        residue.add_atom(atom)
        chain.add_residue(residue)
    gm = gemmi.Model("1")
    gm.add_chain(chain)
    st.add_model(gm)
    st.setup_cell_images()
    return st


def _gemmi_sf(structure, hkl: np.ndarray, addends=None) -> torch.Tensor:
    calc = gemmi.StructureFactorCalculatorX(structure.cell)
    for element, value in (addends or {}).items():
        calc.addends.set(gemmi.Element(element), value)
    return torch.tensor(
        [
            complex(calc.calculate_sf_from_model(structure[0], [int(v) for v in h]))
            for h in hkl
        ],
        dtype=torch.complex128,
    )


def _constant_term_sum(model: ModelFT, hkl: np.ndarray, values: dict) -> torch.Tensor:
    """``sum_mates occ_j c_j T_j exp(2 pi i h.x_j)`` in float64, for c_j = ``values``.

    Every symmetry mate is generated explicitly, ``x' = R x + t``, with the mate's
    temperature factor ``exp(-B s^2 / 4)`` or ``exp(-2 pi^2 s^T U' s)`` from the rotated
    Cartesian ``U' = R_c U R_c^T``. Only atoms whose element is in ``values`` are summed.
    """
    orth = np.array(gemmi.UnitCell(*model.cell.data.tolist()).orth.mat.tolist())
    frac = np.linalg.inv(orth)
    h = np.asarray(hkl, dtype=np.float64)
    s = h @ frac  # Cartesian scattering vectors, row-wise
    s2 = (s * s).sum(axis=1)

    xyz = model.xyz().detach().double().numpy()
    adp = model.adp().detach().double().numpy()
    occ = model.occupancy().detach().double().numpy()
    u = model.u().detach().double().numpy()
    aniso = model.aniso_flag.numpy()
    elements = model.ctx.topology.atoms.element.tolist()
    ops = gemmi.find_spacegroup_by_name(model.spacegroup.hm).operations()

    total = np.zeros(len(h), dtype=np.complex128)
    for op in ops:
        rot = np.array(op.rot, dtype=np.float64) / op.DEN
        tran = np.array(op.tran, dtype=np.float64) / op.DEN
        rot_cart = orth @ rot @ frac
        for i, element in enumerate(elements):
            if element not in values:
                continue
            phase = np.exp(2j * np.pi * (h @ (rot @ (frac @ xyz[i]) + tran)))
            if aniso[i]:
                u11, u22, u33, u12, u13, u23 = u[i]
                U = np.array([[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]])
                U = rot_cart @ U @ rot_cart.T
                dwf = np.exp(-2 * np.pi**2 * np.einsum("ri,ij,rj->r", s, U, s))
            else:
                dwf = np.exp(-adp[i] * s2 / 4.0)
            total += occ[i] * values[element] * dwf * phase
    return torch.from_numpy(total)


def _anomalous_terms(model: ModelFT):
    """``({element: f'}, {element: f''})`` for the elements the model treats as anomalous."""
    significant = get_significant_elements(
        sorted(set(model.ctx.topology.atoms.element.tolist())),
        model.wavelength,
        model.anomalous_threshold,
    )
    assert significant, "no significant anomalous scatterer: the test would be vacuous"
    return (
        {e: fp for e, (fp, _) in significant.items()},
        {e: fdp for e, (_, fdp) in significant.items()},
    )


def _asu_hkl(cell, spacegroup, d_min: float, n: int = N_REFL) -> np.ndarray:
    """``n`` ASU reflections spread evenly over the resolution range to ``d_min``."""
    hkl = gemmi.make_miller_array(cell, spacegroup, d_min)
    hkl = hkl[np.argsort(cell.calculate_d_array(hkl))]
    return hkl[np.linspace(0, len(hkl) - 1, n).round().astype(int)]


def _model_hkl(model: ModelFT, d_min: float, n: int = N_REFL) -> np.ndarray:
    cell = gemmi.UnitCell(*[float(v) for v in model.cell.data.tolist()])
    return _asu_hkl(cell, gemmi.find_spacegroup_by_name(model.spacegroup.hm), d_min, n)


def _F(model: ModelFT, hkl: np.ndarray, **kwargs) -> torch.Tensor:
    with torch.no_grad():
        F = model(torch.tensor(hkl, dtype=torch.int32), recalc=True, **kwargs)
    return F.to(torch.complex128)


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).norm() / ref.norm())


@pytest.fixture(scope="module", params=STRUCTURES)
def scene(request, pdb_dir):
    """One structure's model, reflections ``(asu, -asu)`` and every F the tests compare.

    ``F_f0`` has no anomalous term, ``F_fp`` adds f' and ``F_fdp`` adds f' and f''.
    ``G_f0`` / ``G_fp`` are gemmi's without and with the f' addends.
    """
    code, d_min = request.param
    model = _model(pdb_dir, code, d_min, apply_bijvoet=True)
    st = _gemmi_structure(model)
    f_prime, f_double_prime = _anomalous_terms(model)
    asu = _model_hkl(model, d_min)
    hkl = np.concatenate([asu, -asu])
    d = st.cell.calculate_d_array(hkl)

    F_fdp = _F(model, hkl)
    F_f0 = _F(model, hkl, apply_anomalous=False)
    model.anomalous_bijvoet.fill_(False)
    F_fp = _F(model, hkl)
    model.anomalous_bijvoet.fill_(True)
    return SimpleNamespace(
        code=code,
        model=model,
        hkl=hkl,
        n=len(asu),
        top=torch.from_numpy(d < np.quantile(d, 0.25)),
        f_prime=f_prime,
        f_double_prime=f_double_prime,
        F_f0=F_f0,
        F_fp=F_fp,
        F_fdp=F_fdp,
        G_f0=_gemmi_sf(st, hkl),
        G_fp=_gemmi_sf(st, hkl, f_prime),
    )


@pytest.mark.unit
def test_f_prime_matches_gemmi_addends(scene):
    """f' (Bijvoet off) matches gemmi with addends as closely as f0 alone matches gemmi.

    The top resolution quarter is gated on its own because a dispersive term missing
    the Debye-Waller factor is worst there, and the f' contribution ``F - F(f0)`` is
    gated directly because in the totals it is diluted by the f0 grid error.
    """
    s, top = scene, scene.top
    err_f0, err_fp = _rel(s.F_f0, s.G_f0), _rel(s.F_fp, s.G_fp)
    top_f0, top_fp = _rel(s.F_f0[top], s.G_f0[top]), _rel(s.F_fp[top], s.G_fp[top])
    term = _rel(s.F_fp - s.F_f0, s.G_fp - s.G_f0)
    print(
        f"\n  {s.code} f' {s.f_prime}: rel L2 vs gemmi {err_fp:.3e} (f0 alone "
        f"{err_f0:.3e}); top quarter {top_fp:.3e} ({top_f0:.3e}); f' term {term:.3e}"
    )
    assert err_fp < ERROR_GROWTH * err_f0
    assert top_fp < ERROR_GROWTH * top_f0
    assert term < TERM_TOL


@pytest.mark.unit
def test_f_double_prime_matches_explicit_sum(scene):
    """With f'' on, F, the f'' term and the Bijvoet differences match the reference.

    The reference is gemmi's F with the f' addends plus ``i`` times the explicit f''
    sum. The Bijvoet differences ``|F(h)| - |F(-h)|`` come from f'' alone.
    """
    s, n = scene, scene.n
    # The explicit sum must reproduce gemmi's own f' contribution before it is
    # trusted with f''.
    assert _rel(_constant_term_sum(s.model, s.hkl, s.f_prime), s.G_fp - s.G_f0) < 1e-6
    fdp_ref = _constant_term_sum(s.model, s.hkl, s.f_double_prime)
    ref = s.G_fp + 1j * fdp_ref

    err_f0, err_fdp = _rel(s.F_f0, s.G_f0), _rel(s.F_fdp, ref)
    term = _rel((s.F_fdp - s.F_fp) / 1j, fdp_ref)
    bijvoet = s.F_fdp[:n].abs() - s.F_fdp[n:].abs()
    bijvoet_ref = ref[:n].abs() - ref[n:].abs()
    rms_ref = float(bijvoet_ref.pow(2).mean().sqrt())
    bijvoet_err = float((bijvoet - bijvoet_ref).pow(2).mean().sqrt()) / rms_ref
    cc = float(np.corrcoef(bijvoet.numpy(), bijvoet_ref.numpy())[0, 1])
    print(
        f"\n  {s.code} f'' {s.f_double_prime}: rel L2 {err_fdp:.3e} (f0 alone "
        f"{err_f0:.3e}); f'' term {term:.3e}; Bijvoet differences rms "
        f"{rms_ref:.3f} e, error {bijvoet_err:.2%}, CC {cc:.6f}"
    )
    assert err_fdp < ERROR_GROWTH * err_f0
    assert term < TERM_TOL
    assert bijvoet_err < TERM_TOL
    assert cc > 0.999


@pytest.mark.unit
@pytest.mark.parametrize("code, d_min", [("3E98", 2.5), ("6G9X", 2.3)])
def test_symmetry_equivalents_share_amplitude(pdb_dir, code, d_min):
    """Symmetry mates of h share |F|; Friedel mates share it only without f''.

    Both groups are chiral, so every equivalent ``hR`` is in the same Bijvoet class as
    ``h`` and every ``-hR`` in the other. f'' separates the two classes and nothing
    may separate members of one class.
    """
    model = _model(pdb_dir, code, d_min)
    ops = gemmi.find_spacegroup_by_name(model.spacegroup.hm).operations()
    asu = _model_hkl(model, d_min, n=150)
    mates = np.array(
        [[op.apply_to_hkl([int(v) for v in h]) for op in ops] for h in asu]
    )
    n_refl, n_ops = mates.shape[:2]
    flat = mates.reshape(-1, 3)
    hkl = np.concatenate([flat, -flat])

    for bijvoet in (False, True):
        model.anomalous_bijvoet.fill_(bijvoet)
        amp = _F(model, hkl).abs().reshape(2, n_refl, n_ops)
        scale = float(amp.pow(2).mean().sqrt())
        spread = float((amp.max(dim=2).values - amp.min(dim=2).values).max()) / scale
        friedel = float((amp[0, :, 0] - amp[1, :, 0]).pow(2).mean().sqrt()) / scale
        print(
            f"\n  {code} bijvoet={bijvoet}: worst spread across {n_ops} symmetry "
            f"mates {spread:.2e}, rms Friedel difference {friedel:.2e} (of rms |F|)"
        )
        assert spread < 1e-4
        if bijvoet:
            assert friedel > 100 * max(spread, 1e-7)
        else:
            assert friedel < 1e-4


@pytest.mark.unit
def test_imaginary_density_follows_early_symmetry(pdb_dir):
    """The map-space symmetry path treats the f'' density as the reciprocal one does.

    Without late symmetry both parts of the complex density are symmetrized on the
    grid before the FFT; the result must agree with the reciprocal-space expansion.
    """
    model = _model(pdb_dir, "3E98", 2.5, apply_bijvoet=True)
    asu = _model_hkl(model, 2.5, n=150)
    hkl = np.concatenate([asu, -asu])
    late = _F(model, hkl)
    model.fft.use_late_symmetry = False
    early = _F(model, hkl)
    iso, aniso = model.get_iso(), model.get_aniso()
    assert model._add_anomalous_scattering(iso, aniso, include_fdp=True)[2] is not None
    assert _rel(early, late) < 1e-5


@pytest.mark.unit
def test_f_double_prime_reaches_the_gradient(pdb_dir):
    """A Bijvoet-difference target differentiates through the imaginary density."""
    model = _model(pdb_dir, "3E98", 2.5, apply_bijvoet=True)
    f_prime, _ = _anomalous_terms(model)
    asu = torch.tensor(_model_hkl(model, 2.5, n=150), dtype=torch.int32)
    elements = model.ctx.topology.atoms.element.tolist()
    rows = [i for i, e in enumerate(elements) if e in f_prime]

    def bijvoet_grad_norm(apply_bijvoet):
        model.anomalous_bijvoet.fill_(apply_bijvoet)
        F = model(torch.cat([asu, -asu]), recalc=True)
        loss = (F[: len(asu)].abs() - F[len(asu) :].abs()).pow(2).sum()
        (grad,) = torch.autograd.grad(loss, model.xyz.refinable_params)
        assert torch.isfinite(grad).all()
        return float(grad[rows].norm())

    # Without f'' the Bijvoet differences are float32 noise, and so is their gradient.
    assert bijvoet_grad_norm(True) > 100 * bijvoet_grad_norm(False)


@pytest.mark.unit
def test_wavelength_none_is_the_f0_path(pdb_dir):
    """``wavelength=None`` gives exactly the f0 transform, and so does a wavelength
    at which no element is significant."""
    model = _model(pdb_dir, "3E98", 2.5, wavelength=None)
    hkl = torch.tensor(_model_hkl(model, 2.5, n=150), dtype=torch.int32)

    with torch.no_grad():
        F_none = model(hkl, recalc=True)
        F_f0, _ = model.fft.compute_structure_factors(
            hkl, *model.get_iso(), *model.get_aniso(), apply_symmetry=True
        )
        F_off = _model(pdb_dir, "3E98", 2.5)(hkl, recalc=True, apply_anomalous=False)
    assert torch.equal(F_none, F_f0)
    assert torch.equal(F_none, F_off)

    # 1DAW: Mg, P and S stay below the 0.5 e threshold at 1 A.
    light = _model(pdb_dir, "1DAW", 2.5, apply_bijvoet=True)
    assert light._get_anomalous_cache() is None
    hkl = torch.tensor([[1, 2, 3], [-1, -2, -3], [4, 0, 2]], dtype=torch.int32)
    with torch.no_grad():
        assert torch.equal(
            light(hkl, recalc=True), light(hkl, recalc=True, apply_anomalous=False)
        )
