"""Unit tests for the BAOAB Langevin integrator (guided-MD core).

These cover the physical-mass seeding path added for thermostatted ensemble
MD: ``set_physical_masses`` plus the thermostat's equipartition behaviour with
those masses. No crystallography / OpenMM needed — pure integrator math.
"""

import math

import torch

from torchref.refinement.optimizers.langevin_sa import LangevinSA


def _make(param, **kw):
    """LangevinSA with adaptive masses OFF (physical-mass mode) at constant T."""
    defaults = dict(
        dt=0.01,
        friction=10.0,
        T_initial=2.494,
        T_final=2.494,  # constant bath
        total_steps=1000,
        adaptive_masses=False,
        max_step_size=0.5,
    )
    defaults.update(kw)
    return LangevinSA([param], **defaults)


def test_set_physical_masses_seeds_state():
    torch.manual_seed(0)
    p = torch.zeros(5, 3, requires_grad=True)
    opt = _make(p)
    mass = torch.full((5, 1), 12.0)  # (N,1) broadcasts over xyz
    opt.set_physical_masses({id(p): mass}, T=2.494)

    state = opt.state[p]
    assert state["mass"].shape == (5, 3)
    assert torch.allclose(state["mass"], torch.full((5, 3), 12.0))
    assert state["velocity"].shape == (5, 3)
    assert torch.isfinite(state["velocity"]).all()
    assert state["prev_grad"] is None


def test_set_physical_masses_unit_fallback():
    """A param absent from the mass dict keeps unit mass (mass=None)."""
    torch.manual_seed(0)
    p = torch.zeros(4, 3, requires_grad=True)
    opt = _make(p)
    opt.set_physical_masses({}, T=1.0)
    state = opt.state[p]
    assert state["mass"] is None
    assert torch.isfinite(state["velocity"]).all()


def test_baoab_step_moves_and_finite():
    """One BAOAB step on a harmonic well moves params and stays finite."""
    torch.manual_seed(1)
    p = torch.randn(6, 3, requires_grad=True)
    x0 = p.detach().clone()
    opt = _make(p)
    opt.set_physical_masses({id(p): torch.full((6, 1), 12.0)}, T=2.494)

    def closure():
        if p.grad is not None:
            p.grad = None
        loss = 0.5 * (p ** 2).sum()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert torch.isfinite(loss)
    assert torch.isfinite(p).all()
    assert not torch.allclose(p.detach(), x0)  # it moved


def test_baoab_equipartition_free_particle():
    """Free particle (U=0): the thermostat equilibrates KE to ½·dof·T.

    With physical masses, the O-step stationary velocity variance is T/m, so
    KE = Σ ½ m v² → ½·N_dof·T independent of mass. Validates that the thermal
    noise is correctly mass/T-scaled (the property that decouples it from the
    gradient).
    """
    torch.manual_seed(2)
    n_atoms = 2000
    T = 2.494
    p = torch.zeros(n_atoms, 3, requires_grad=True)
    opt = _make(p, friction=20.0, dt=0.01)
    # Heterogeneous masses (H..S range) to exercise the per-element scaling.
    mass = torch.empty(n_atoms, 1).uniform_(1.0, 32.0)
    opt.set_physical_masses({id(p): mass}, T=T)

    def closure():
        if p.grad is not None:
            p.grad = None
        loss = (p * 0.0).sum()  # U = 0 → zero force
        loss.backward()
        return loss

    # Burn in, then average KE over a window to cut sampling noise.
    for _ in range(200):
        opt.step(closure)
    kes = []
    for _ in range(200):
        opt.step(closure)
        kes.append(opt.kinetic_energy)

    n_dof = n_atoms * 3
    expected = 0.5 * n_dof * T
    measured = sum(kes) / len(kes)
    rel = abs(measured - expected) / expected
    assert rel < 0.1, f"equipartition off: measured {measured:.1f} vs {expected:.1f} (rel {rel:.3f})"
