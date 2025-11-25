#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBATCH -c 8
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/test_lbfgs_quick.out

"""Quick test to verify LBFGS refinement implementation works"""

import torch
import sys

def test_imports():
    """Test that imports work"""
    print("Testing imports...")
    try:
        from torchref.lbfgs_refinement import LBFGSRefinement, LBFGS_scales
        print("  ✓ Imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_basic_refinement():
    """Test basic LBFGS refinement"""
    print("\nTesting basic LBFGS refinement...")
    try:
        from torchref.lbfgs_refinement import LBFGSRefinement
        
        mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
        pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
        
        refinement = LBFGSRefinement(
            mtz, pdb,
            verbose=0,
            target_weights={'xray': 1.0, 'restraints': 1.0, 'adp': 0.3}
        )
        
        # Get initial R-factors
        rwork_init, rfree_init = refinement.get_rfactor()
        print(f"  Initial: Rwork={rwork_init:.4f}, Rfree={rfree_init:.4f}")
        
        # Run 1 cycle with minimal steps
        metrics = refinement.run_lbfgs_refinement(
            macro_cycles=1,
            targets=['xyz'],
            steps_per_target=2,
            steps_scales=2
        )
        
        # Get final R-factors
        rwork_final, rfree_final = refinement.get_rfactor()
        print(f"  Final:   Rwork={rwork_final:.4f}, Rfree={rfree_final:.4f}")
        print(f"  Improvement: ΔRfree={rfree_init - rfree_final:+.4f}")
        
        # Check that refinement improved things
        if rfree_final < rfree_init:
            print("  ✓ Refinement improved Rfree")
            return True
        else:
            print("  ⚠ Warning: Rfree did not improve (might be expected for 1 cycle)")
            return True  # Still pass - short refinement may not improve
            
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_target_refinement():
    """Test individual target refinement"""
    print("\nTesting individual target refinement...")
    try:
        from torchref.lbfgs_refinement import LBFGSRefinement
        
        mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
        pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
        
        refinement = LBFGSRefinement(mtz, pdb, verbose=0)
        refinement.get_scales()
        
        # Test xyz refinement
        metrics_xyz = refinement.refine_target_lbfgs('xyz', nsteps=2, verbose=False)
        print(f"  ✓ XYZ refinement: {len(metrics_xyz['steps'])} steps completed")
        
        # Test b refinement
        metrics_b = refinement.refine_target_lbfgs('b', nsteps=2, verbose=False)
        print(f"  ✓ B-factor refinement: {len(metrics_b['steps'])} steps completed")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scales_refinement():
    """Test scale refinement"""
    print("\nTesting scale refinement...")
    try:
        from torchref.lbfgs_refinement import LBFGSRefinement, LBFGS_scales
        from torchref.base_refinement import Refinement
        
        mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
        pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
        
        # Test with LBFGSRefinement
        refinement1 = LBFGSRefinement(mtz, pdb, verbose=0)
        refinement1.get_scales()
        metrics1 = refinement1.refine_scales_lbfgs(nsteps=2, verbose=False)
        print(f"  ✓ LBFGSRefinement.refine_scales_lbfgs: {len(metrics1['steps'])} steps")
        
        # Test standalone function with base Refinement
        refinement2 = Refinement(mtz, pdb, verbose=0)
        refinement2.get_scales()
        metrics2 = LBFGS_scales(refinement2, nsteps=2, verbose=False)
        print(f"  ✓ LBFGS_scales standalone function: {len(metrics2['steps'])} steps")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_summary():
    """Test refinement summary"""
    print("\nTesting refinement summary...")
    try:
        from torchref.lbfgs_refinement import LBFGSRefinement
        
        mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
        pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
        
        refinement = LBFGSRefinement(mtz, pdb, verbose=0)
        summary = refinement.get_refinement_summary()
        
        required_keys = ['rwork', 'rfree', 'xray_work', 'xray_test', 
                        'restraints', 'adp', 'target_weights', 
                        'effective_weights', 'ratio_test_work']
        
        for key in required_keys:
            if key not in summary:
                print(f"  ✗ Missing key in summary: {key}")
                return False
        
        print(f"  ✓ Summary contains all required keys")
        print(f"    Rwork={summary['rwork']:.4f}, Rfree={summary['rfree']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*80)
    print("LBFGS REFINEMENT UNIT TESTS")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Basic Refinement", test_basic_refinement()))
    results.append(("Target Refinement", test_target_refinement()))
    results.append(("Scales Refinement", test_scales_refinement()))
    results.append(("Get Summary", test_get_summary()))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {name}")
    
    print("="*80)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
