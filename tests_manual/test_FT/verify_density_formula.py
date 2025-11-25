"""
Verify the density at atom center matches theory.
"""
import numpy as np
import torch
import gemmi
from torchref.model_ft import ModelFT

print("=" * 70)
print("VERIFY DENSITY AT ATOM CENTER")
print("=" * 70)

# Parameters
element = "C"
B_atom = 23.26  # From the test atom (C VAL 14)

fe = gemmi.Element(element).it92
a = np.array(fe.a)
b = np.array(fe.b)

print(f"Element: {element}")
print(f"B-factor: {B_atom} Ų")
print(f"ITC92 a: {a}")
print(f"ITC92 b: {b}")

# The CORRECT formula from Fourier transform:
# Reciprocal space: f(s) = Σ aᵢ exp(-(bᵢ + B)s²)
# Real space: ρ(r) = Σ aᵢ (π/(bᵢ+B))^(3/2) exp(-π²r²/(bᵢ+B))
# At r=0: ρ(0) = Σ aᵢ (π/(bᵢ+B))^(3/2)

print("\n" + "=" * 70)
print("Theory: Density at r=0")
print("=" * 70)

density_theory = 0.0
for i in range(4):
    B_total = b[i] + B_atom
    contribution = a[i] * (np.pi / B_total) ** 1.5
    print(f"Component {i}: a={a[i]:.4f}, b={b[i]:.4f}, B_tot={B_total:.4f}")
    print(f"  Contribution: {contribution:.6f} e/Ų")
    density_theory += contribution

print(f"\nTotal density at r=0 (theory): {density_theory:.6f} e/Ų")

# Now check our implementation
print("\n" + "=" * 70)
print("Our implementation:")
print("=" * 70)

M = ModelFT()
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark.pdb')
M.max_res = 1.2
M.setup_grid()
M.build_density_map(apply_symmetry=False)

# Find the C VAL 14 atom
test_atom_idx = 100
row = M.pdb.iloc[test_atom_idx]
test_pos = np.array([row['x'], row['y'], row['z']])

print(f"Test atom: {row['name']} {row['resname']} {row['resseq']}")
print(f"  B-factor: {row['tempfactor']:.2f} Ų")
print(f"  Position: {test_pos}")

# Get grid position
atom_cart = torch.tensor(test_pos, dtype=torch.float64)
atom_frac = M.inv_frac_matrix @ atom_cart
atom_frac_wrapped = torch.remainder(atom_frac, 1.0)
grid_idx = (atom_frac_wrapped * torch.tensor(M.map.shape, dtype=torch.float64)).long()

ix, iy, iz = grid_idx
density_actual = M.map[ix, iy, iz].item()

print(f"\nDensity from map: {density_actual:.6f} e/Ų")
print(f"Density from theory: {density_theory:.6f} e/Ų")
print(f"Ratio (actual/theory): {density_actual/density_theory:.4f}")

print("\n" + "=" * 70)
print("INTERPRETATION:")
print("=" * 70)
if abs(density_actual/density_theory - 1.0) < 0.1:
    print("✓ EXCELLENT! Ratio ≈ 1.0 - formula is correct!")
elif abs(density_actual/density_theory - 2.0) < 0.2:
    print("✗ Ratio ≈ 2.0 - B-factor might be 2× too small in formula")
elif abs(density_actual/density_theory - 0.5) < 0.1:
    print("✗ Ratio ≈ 0.5 - B-factor might be 2× too large in formula")
else:
    print(f"? Unexpected ratio: {density_actual/density_theory:.4f}")
    print("  May need to check:")
    print("  - Voxel not exactly at atom center")
    print("  - Interpolation effects")
    print("  - Numerical precision")

# Check distance from atom center to nearest voxel
atom_cart_wrapped = M.frac_matrix @ atom_frac_wrapped
voxel_cart = M.real_space_grid[ix, iy, iz]
distance = torch.norm(atom_cart_wrapped - voxel_cart).item()
print(f"\nDistance from atom to voxel center: {distance:.4f} Å")
print(f"Voxel size: ~{M.voxel_size.mean().item():.4f} Å")

if distance > 0.1:
    print("NOTE: Atom not at voxel center - density will be lower!")
    # Estimate correction
    density_corrected = density_theory * np.exp(-np.pi**2 * distance**2 / (b.mean() + B_atom))
    print(f"  Density at distance {distance:.4f} Å (estimated): {density_corrected:.6f} e/Ų")
    print(f"  Corrected ratio: {density_actual/density_corrected:.4f}")
