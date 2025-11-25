"""
Test that Model evaluates to its initialized attribute in boolean context.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchref.model import Model

print("Testing Model.__bool__() method...")
print("=" * 60)

# Create a new model (not initialized)
model = Model()
print(f"\n1. After creating Model():")
print(f"   model.initialized = {model.initialized}")
print(f"   bool(model) = {bool(model)}")
print(f"   if model: evaluates to {True if model else False}")

# Test in if statement
if model:
    print("   ✗ ERROR: Model evaluated to True when not initialized!")
    sys.exit(1)
else:
    print("   ✓ Correctly evaluated to False (not initialized)")

# Manually set initialized to True
model.initialized = True
print(f"\n2. After setting model.initialized = True:")
print(f"   model.initialized = {model.initialized}")
print(f"   bool(model) = {bool(model)}")
print(f"   if model: evaluates to {True if model else False}")

# Test in if statement
if model:
    print("   ✓ Correctly evaluated to True (initialized)")
else:
    print("   ✗ ERROR: Model evaluated to False when initialized!")
    sys.exit(1)

# Create another model and load data (if we have test data)
print(f"\n3. Testing with actual initialization:")
model2 = Model()
print(f"   Before load: bool(model2) = {bool(model2)}")

# Note: We can't actually load a PDB without a file, but we can test the pattern
print(f"   model2.initialized would be set to True after load_pdb_from_file()")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("\nYou can now use:")
print("  if model:")
print("      # Model is initialized")
print("  else:")
print("      # Model is not initialized")
