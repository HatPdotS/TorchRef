"""
Convert dark (8QL2) and light (torchref_0p18) PDB structures into an
IHM mmCIF ensemble file with embedded reflection data.

Creates a MixedModel with two states (dark / light) and writes it as
an IHM-compliant mmCIF file with proper multi-state annotations,
population fractions, and per-timepoint reflection data.

Usage:
    python create_ihm_ensemble.py
"""

from pathlib import Path

import torch

from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model.model_ft import ModelFT
from torchref.model.mixed_model import MixedModel

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DARK_PDB = DATA_DIR / "8QL2.pdb"
LIGHT_PDB = DATA_DIR / "torchref_0p18.pdb"
DARK_MTZ = DATA_DIR / "dark-phenix.mtz"
LIGHT_MTZ = DATA_DIR / "7YYZ-light.mtz"
OUTPUT_CIF = DATA_DIR / "dark_light_ensemble.cif"

# ── Population fractions (dark, light) ─────────────────────────────────
# 82% dark / 18% light matches the 0.18 photoactivation yield
FRACTIONS = [0.82, 0.18]

# ── Load models ────────────────────────────────────────────────────────
print(f"Loading dark model:  {DARK_PDB.name}")
model_dark = ModelFT(max_res=2.0, verbose=0)
model_dark.load_pdb(str(DARK_PDB))

print(f"Loading light model: {LIGHT_PDB.name}")
model_light = ModelFT(max_res=2.0, verbose=0)
model_light.load_pdb(str(LIGHT_PDB))

print(f"  Dark atoms:  {len(model_dark.pdb)}")
print(f"  Light atoms: {len(model_light.pdb)}")

# ── Load reflection data ─────────────────────────────────────────────
print(f"\nLoading dark data:   {DARK_MTZ.name}")
dark_data = ReflectionData()
dark_data.load_mtz(str(DARK_MTZ))
print(f"  Reflections: {len(dark_data.hkl)}")

print(f"Loading light data:  {LIGHT_MTZ.name}")
light_data = ReflectionData()
light_data.load_mtz(str(LIGHT_MTZ))
print(f"  Reflections: {len(light_data.hkl)}")

# ── Build MixedModel ──────────────────────────────────────────────────
mixed = MixedModel(
    [model_dark, model_light],
    initial_fractions=FRACTIONS,
    frozen_fractions=True,
)
print(f"\n{mixed}")

# ── Write IHM mmCIF ───────────────────────────────────────────────────
print(f"\nWriting IHM ensemble to: {OUTPUT_CIF.name}")
mixed.write_ihm(
    str(OUTPUT_CIF),
    state_names=["dark_state", "light_state"],
    group_name="photoactivated_mixture",
    datasets={"dark": dark_data, "light": light_data},
)
print("Done.")
