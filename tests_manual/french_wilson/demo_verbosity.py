"""
Demonstration of verbosity levels in FrenchWilsonModule.

This script shows the different verbosity levels and their output.
"""

import torch
import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.french_wilson import FrenchWilsonModule


def demo_verbosity_levels():
    """Demonstrate the different verbosity levels"""
    
    # Example data
    hkl = torch.tensor([[1, 2, 3], [2, 0, 0], [0, 3, 0], [1, 1, 1], [3, 2, 1]])
    unit_cell = [50.0, 60.0, 70.0, 90.0, 90.0, 90.0]
    
    print("=" * 70)
    print("FrenchWilsonModule Verbosity Levels Demo")
    print("=" * 70)
    
    # verbose=0: Silent (no output)
    print("\n" + "=" * 70)
    print("VERBOSE = 0: Silent mode (no initialization messages)")
    print("=" * 70)
    fw0 = FrenchWilsonModule(hkl, unit_cell, 'P212121', verbose=0)
    print("(No output - useful for production code or batch processing)")
    
    # verbose=1: Basic info (DEFAULT)
    print("\n" + "=" * 70)
    print("VERBOSE = 1: Basic info (DEFAULT - recommended for most users)")
    print("=" * 70)
    fw1 = FrenchWilsonModule(hkl, unit_cell, 'P212121', verbose=1)
    print("\nShows: number of reflections, resolution range, space group, centric %")
    
    # verbose=2: Detailed info
    print("\n" + "=" * 70)
    print("VERBOSE = 2: Detailed info (for debugging/development)")
    print("=" * 70)
    fw2 = FrenchWilsonModule(hkl, unit_cell, 'P212121', verbose=2)
    print("\nAdditional info: binning parameters, rejection thresholds, device")
    
    # Usage examples
    print("\n" + "=" * 70)
    print("Usage Examples")
    print("=" * 70)
    
    print("\n# For production code (silent):")
    print("fw = FrenchWilsonModule(hkl, unit_cell, space_group, verbose=0)")
    
    print("\n# For interactive use (default):")
    print("fw = FrenchWilsonModule(hkl, unit_cell, space_group)")
    print("# or explicitly:")
    print("fw = FrenchWilsonModule(hkl, unit_cell, space_group, verbose=1)")
    
    print("\n# For debugging:")
    print("fw = FrenchWilsonModule(hkl, unit_cell, space_group, verbose=2)")
    
    print("\n" + "=" * 70)
    print("Verbosity Levels Summary")
    print("=" * 70)
    print("0 = Silent       (no output)")
    print("1 = Basic info   (DEFAULT - initialization summary)")
    print("2 = Detailed     (additional debug information)")
    print("=" * 70)


if __name__ == "__main__":
    demo_verbosity_levels()
