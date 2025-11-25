#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
"""Quick debug test"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.scaler_analytical import AnalyticalScaler
import inspect

# Check the forward method signature
print("Forward method signature:")
sig = inspect.signature(AnalyticalScaler.forward)
print(f"  {sig}")
print(f"  Parameters: {list(sig.parameters.keys())}")

# Check if it's callable
print(f"\nAnalyticalScaler.forward is callable: {callable(AnalyticalScaler.forward)}")
print(f"Type: {type(AnalyticalScaler.forward)}")
