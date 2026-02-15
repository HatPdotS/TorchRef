"""
Compare standard LBFGS vs Tunneling LBFGS on a rugged 1D landscape.
"""

import torch
from tunneling_lbfgs import TunnelingLBFGS


def hard_to_optimize(x):
    return torch.sin(5 * (x - 4)) + 0.01 * (x - 4) ** 2


def run_standard_lbfgs(x0_val, n_steps=100):
    x = torch.tensor(x0_val, requires_grad=True)
    opt = torch.optim.LBFGS([x], max_iter=100, line_search_fn="strong_wolfe")
    trajectory = [x0_val]

    def closure():
        opt.zero_grad()
        loss = hard_to_optimize(x)
        loss.backward()
        return loss

    for _ in range(n_steps):
        opt.step(closure)
        trajectory.append(x.item())

    return x, trajectory


def run_tunneling_lbfgs(x0_val, n_steps=100, **kwargs):
    x = torch.tensor(x0_val, requires_grad=True)
    opt = TunnelingLBFGS([x], **kwargs)
    trajectory = [x0_val]

    def closure():
        opt.zero_grad()
        loss = hard_to_optimize(x)
        loss.backward()
        return loss

    for _ in range(n_steps):
        opt.step(closure)
        trajectory.append(x.item())

    return x, trajectory


# ---- run both ----
print("=" * 60)
print("Standard LBFGS (Strong Wolfe)")
print("=" * 60)
x_std, traj_std = run_standard_lbfgs(0.0)
print(f"  x = {x_std.item():.6f},  f(x) = {hard_to_optimize(x_std).item():.6f}")

print()
print("=" * 60)
print("Tunneling LBFGS (over-exploring line search)")
print("=" * 60)
x_tun, traj_tun = run_tunneling_lbfgs(
    0.0,
    n_steps=30,
    lr=1.0,
    max_step=10.0,
    n_scan_coarse=64,
    n_scan_fine=16,
    scan_spacing="log",
)
print(f"  x = {x_tun.item():.6f},  f(x) = {hard_to_optimize(x_tun).item():.6f}")

# ---- ground truth ----
xs = torch.linspace(-10, 10, 100000)
ys = hard_to_optimize(xs)
idx = ys.argmin()
print(f"\n  Global min: x = {xs[idx].item():.6f},  f(x) = {ys[idx].item():.6f}")

# ---- plot ----
from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_grid = torch.linspace(-10, 10, 2000)
y_grid = hard_to_optimize(x_grid)

for ax, title, traj, x_final in [
    (axes[0], "Standard LBFGS (Strong Wolfe)", traj_std, x_std),
    (axes[1], "Tunneling LBFGS", traj_tun, x_tun),
]:
    ax.plot(x_grid.numpy(), y_grid.numpy(), "b-", alpha=0.5, lw=1)
    # trajectory
    traj_y = [hard_to_optimize(torch.tensor(xi)).item() for xi in traj]
    ax.plot(traj, traj_y, "ro-", markersize=3, lw=0.8, label="trajectory")
    # start & end
    ax.plot(traj[0], traj_y[0], "gs", markersize=10, label="start")
    ax.plot(
        x_final.item(),
        hard_to_optimize(x_final).item(),
        "r*",
        markersize=15,
        label=f"final x={x_final.item():.2f}",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=8)
    ax.set_ylim(-1.5, 2.5)

plt.tight_layout()
plt.savefig(
    "/das/work/p17/p17490/Peter/Library/torchref/tests/LBFGS_optimizers/tunneling_comparison.png",
    dpi=150,
)
print("\nPlot saved to tunneling_comparison.png")
