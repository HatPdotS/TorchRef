"""The density splat operators are well-formed PyTorch operators on every device they ship on.

``torch.library.opcheck`` checks what the dispatcher and ``torch.compile`` rely on and the
numerics tests cannot see: the schema matches what the kernels do (no undeclared mutation
or aliasing), the fake kernels agree with the real ones on shape/dtype/device, and the ops
trace under AOT dispatch. Autograd is not the operators' -- it is :class:`SplatIso` /
:class:`SplatAniso`'s, covered by the gradient and second-order tests -- so the inputs
here do not require grad.
"""

import pytest
import torch

from torchref.base.electron_density import ops
from torchref.base.electron_density.kernels.cpu import sphere_splat

pytestmark = pytest.mark.unit


def _inputs(device, dtype, aniso, n=12, grid=(10, 12, 9)):
    g = torch.Generator().manual_seed(3)
    frac = torch.tensor([[12.0, 0.0, -1.5], [0.0, 14.0, 0.0], [0.0, 0.0, 11.0]],
                        dtype=torch.float64)
    inv = frac.inverse()
    xyz = torch.rand(n, 3, generator=g, dtype=torch.float64) @ frac.T
    adp = (0.3 + 0.2 * torch.rand(n, 3, generator=g, dtype=torch.float64))
    third = torch.cat([adp, torch.zeros(n, 3, dtype=torch.float64)], 1) if aniso else 20 * adp[:, 0]
    occ = 0.5 + 0.5 * torch.rand(n, generator=g, dtype=torch.float64)
    A = 0.5 + torch.rand(n, 5, generator=g, dtype=torch.float64)
    B = 1.0 + 20.0 * torch.rand(n, 5, generator=g, dtype=torch.float64)
    r2cut = torch.full((n,), 2.5 ** 2, dtype=torch.float64)
    grid_t = torch.zeros(grid, dtype=torch.float64)
    to = lambda t: t.to(device=device, dtype=dtype)  # noqa: E731
    return [to(t) for t in (grid_t, xyz, third, occ, A, B, r2cut, inv, frac)]


def _cases():
    cases = [("cpu", torch.float32), ("cpu", torch.float64)]  # dtype-ok: test matrix
    if torch.backends.mps.is_available():
        cases.append(("mps", torch.float32))  # dtype-ok: Metal is float32 only
    return cases


@pytest.mark.parametrize("aniso", [False, True], ids=["iso", "aniso"])
@pytest.mark.parametrize("device,dtype", _cases(), ids=lambda v: str(v).replace("torch.", ""))
def test_splat_ops_pass_opcheck(device, dtype, aniso):
    if device == "cpu" and sphere_splat.why_unavailable() is not None:
        pytest.skip(sphere_splat.why_unavailable())
    fwd = ops.splat_aniso_fwd if aniso else ops.splat_iso_fwd
    bwd = ops.splat_aniso_bwd if aniso else ops.splat_iso_bwd
    args = _inputs(device, dtype, aniso)
    torch.library.opcheck(fwd, args)
    grad = torch.rand(args[0].shape, generator=torch.Generator().manual_seed(1)).to(args[0])
    torch.library.opcheck(bwd, [grad] + args[1:])


def test_ops_dispatch_to_the_inputs_device_kernel(monkeypatch):
    """The dispatcher, not TorchRef, picks the device kernel; prove it is the CPU one."""
    if sphere_splat.why_unavailable() is not None:
        pytest.skip(sphere_splat.why_unavailable())
    seen = []
    original = sphere_splat.iso_fwd
    monkeypatch.setattr(
        sphere_splat, "iso_fwd", lambda *a: seen.append(a[0].device) or original(*a)
    )
    ops.splat_iso_fwd(*_inputs("cpu", torch.float32, False))
    assert seen == [torch.device("cpu")]
