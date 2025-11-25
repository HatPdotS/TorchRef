"""
Debug B-factor convention - compare with CCTBX calculated structure factors.
"""
import torch
import numpy as np
from torchref.model_ft import ModelFT
from torchref.math_functions.math_torch import vectorized_add_to_map
import gemmi

print("=" * 70)
print("B-FACTOR CONVENTION TEST")
print("=" * 70)

# Load structure
M = ModelFT()   
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark.pdb')
M.max_res = 1.2
M.setup_grid()

print(f"\nStructure info:")
print(f"  Number of atoms: {len(M.pdb)}")
print(f"  Map shape: {M.map.shape}")

# Check B-factor values in the PDB
print("\n" + "=" * 70)
print("B-factor statistics from PDB:")
print("=" * 70)
b_factors = M.pdb['tempfactor'].values
print(f"  Min B-factor: {b_factors.min():.2f} Ų")
print(f"  Max B-factor: {b_factors.max():.2f} Ų")
print(f"  Mean B-factor: {b_factors.mean():.2f} Ų")
print(f"  Median B-factor: {np.median(b_factors):.2f} Ų")

# Sample a few atoms
print("\nSample atoms:")
for i in [0, 100, 500]:
    row = M.pdb.iloc[i]
    print(f"  {row['name']:4s} {row['resname']:3s} {row['resseq']:4d}: B = {row['tempfactor']:6.2f} Ų")

# Build density map
print("\n" + "=" * 70)
print("Building density map...")
print("=" * 70)
M.build_density_map(apply_symmetry=False)
print(f"Map sum: {M.map.sum():.2f}")
print(f"Map max: {M.map.max():.4f}")

# Compare with a single atom calculation using gemmi's ITC92 coefficients
print("\n" + "=" * 70)
print("Test: Single atom density calculation")
print("=" * 70)

# Pick an atom with typical B-factor
test_atom_idx = 100
row = M.pdb.iloc[test_atom_idx]
test_b = row['tempfactor']
test_element = row['element']
test_pos = np.array([row['x'], row['y'], row['z']])

print(f"\nTest atom: {row['name']} {row['resname']} {row['resseq']}")
print(f"  Element: {test_element}")
print(f"  B-factor: {test_b:.2f} Ų")
print(f"  Position: {test_pos}")

# Get ITC92 coefficients from gemmi
fe = gemmi.Element(test_element).it92
A_gemmi = np.array(fe.a)
B_gemmi = np.array(fe.b)
C_gemmi = fe.c

print(f"\nITC92 coefficients (gemmi):")
print(f"  A: {A_gemmi}")
print(f"  B: {B_gemmi}")
print(f"  C: {C_gemmi}")

# Calculate density at the atom center using the formula:
# ρ(r) = Σ A_i * exp(-B_i * s² - B_iso/(4π²) * s²)
# where s = 1/d = |r|/(2π)
# At r=0: s=0, so ρ(0) = Σ A_i * exp(0) = Σ A_i

# But we need to account for the B-factor properly
# The ITC92 formula is: f(s) = Σ A_i * exp(-B_i * s²) + C
# For electron density in real space with B-factor:
# ρ(r) = Σ [A_i / (B_tot)^(3/2)] * exp(-π/B_tot * r²)
# where B_tot = B_ITC92 + B_iso

# The Gaussian width parameter in 3D:
# For a Gaussian exp(-α*r²), the normalization constant is (α/π)^(3/2)
# With B = 8π²σ², we have α = 2π²/B, so normalization is (2π²/B / π)^(3/2) = (2π/B)^(3/2)

# At atom center (r=0), density should be:
print(f"\nDensity at atom center (theory):")

# Method 1: Direct from ITC92 A coefficients (normalized)
density_center_theory = 0.0
for i in range(4):
    B_total = B_gemmi[i] + test_b
    # Normalization for 3D Gaussian: (2π²/B)^(3/2) / (2π)^(3/2) = (π/B)^(3/2)
    norm = (np.pi / B_total) ** 1.5
    density_center_theory += A_gemmi[i] * norm

print(f"  Theory (normalized): {density_center_theory:.6f} e⁻/ų")

# Now check what our code calculates
print("\n" + "=" * 70)
print("Checking our implementation:")
print("=" * 70)

# Get the parametrization used in the code
param = M.parametrization[test_element]
A_code = param[0].numpy()[0]
B_code = param[1].numpy()[0]
C_code = param[2].numpy()[0]

print(f"\nOur parametrization for {test_element}:")
print(f"  A: {A_code}")
print(f"  B: {B_code}")
print(f"  C: {C_code}")

# Check if they match
print(f"\nCoefficient comparison:")
print(f"  A match: {np.allclose(A_code, A_gemmi)}")
print(f"  B match: {np.allclose(B_code, B_gemmi)}")

# Now calculate what density we actually get at the atom position
# Convert atom position to fractional
atom_cart = torch.tensor(test_pos, dtype=torch.float64)
atom_frac = M.inv_frac_matrix @ atom_cart

# Wrap to unit cell
atom_frac_wrapped = torch.remainder(atom_frac, 1.0)
atom_cart_wrapped = M.frac_matrix @ atom_frac_wrapped

# Find grid index
grid_idx = (atom_frac_wrapped * torch.tensor(M.map.shape, dtype=torch.float64)).long()

print(f"\nAtom grid position:")
print(f"  Fractional: {atom_frac_wrapped}")
print(f"  Grid index: {grid_idx}")
print(f"  Cartesian (wrapped): {atom_cart_wrapped}")

# Get density at this position
ix, iy, iz = grid_idx
if 0 <= ix < M.map.shape[0] and 0 <= iy < M.map.shape[1] and 0 <= iz < M.map.shape[2]:
    density_at_atom = M.map[ix, iy, iz].item()
    print(f"  Density from map: {density_at_atom:.6f} e⁻/ų")
    
    # Compare with theory
    ratio = density_at_atom / density_center_theory
    print(f"\nRatio (actual/theory): {ratio:.4f}")
    print(f"  If ratio ≈ 0.5, B-factor might be 2× too large")
    print(f"  If ratio ≈ 2.0, B-factor might be 2× too small")
    print(f"  If ratio ≈ 1.0, B-factor is correct")
else:
    print(f"  ERROR: Atom position outside grid!")

# Let's also check the formula used in vectorized_add_to_map
print("\n" + "=" * 70)
print("Checking the Gaussian formula:")
print("=" * 70)

print(f"\nThe code uses:")
print(f"  u_gaussian = 1.0 / (B_ITC92 + B_iso)")
print(f"  exponent = -2*π² * r² * u_gaussian")
print(f"  density = Σ A_i * exp(exponent)")

print(f"\nStandard crystallography convention:")
print(f"  Debye-Waller factor: exp(-B * s²) where s = sin(θ)/λ")
print(f"  In real space: exp(-B/(4π²) * (2π*s)²) = exp(-B * s²)")
print(f"  For Gaussian: exp(-2π² * r² / σ²)")
print(f"  With B = 8π² * σ²: exp(-2π² * r² / (B/(8π²))) = exp(-16π⁴ * r² / B)")

print(f"\nPotential issue:")
print(f"  Our formula: exp(-2π² * r² / B)")
print(f"  Should be: exp(-2π² * r² / (B/(8π²))) = exp(-16π⁴ * r² / B)")
print(f"  Difference: factor of 8π² ≈ 78.96 in the denominator")
print(f"  OR: B definition difference (some use B, some use B/2)")

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)
print("If density is too sharp (high peak, narrow width):")
print("  → B-factor is effectively too small")
print("  → Check if we need: u = 1/(B + B_ITC92) → u = 1/(B/2 + B_ITC92)")
print("\nIf density is too broad (low peak, wide width):")
print("  → B-factor is effectively too large") 
print("  → Check if we need: u = 1/(B + B_ITC92) → u = 1/(2*B + B_ITC92)")
