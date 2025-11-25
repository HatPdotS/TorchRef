#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBATCH -c 4
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/test_quick.out

"""Quick test to verify optimizer setup works before running full benchmark"""

import torch
import time
from torchref.base_refinement import Refinement


def loss_function(refinement):
    loss = 0.0
    if 'xray' in refinement.target_weights:
        loss = loss + refinement.target_weights['xray'] * refinement.xray_loss()
    if 'restraints' in refinement.target_weights:
        loss = loss + refinement.target_weights['restraints'] * refinement.restraints_loss()
    if 'adp' in refinement.target_weights:
        loss = loss + refinement.target_weights['adp'] * refinement.adp_loss()
    return loss


def quick_test_lbfgs(refinement, target, nsteps=2):
    """Quick LBFGS test"""
    print(f"\nTesting LBFGS with target={target}, nsteps={nsteps}")
    refinement.model.freeze_all()
    refinement.model.unfreeze(target)
    refinement.get_scales()
    
    optimizer = torch.optim.LBFGS(refinement.parameters(), lr=1.0)
    
    def closure():
        optimizer.zero_grad()
        loss = loss_function(refinement)
        loss.backward(retain_graph=True)
        return loss
    
    for i in range(nsteps):
        optimizer.step(closure)
        with torch.no_grad():
            xray_work, xray_test = refinement.nll_xray()
            rwork, rfree = refinement.get_rfactor()
            print(f"  Step {i+1}: X-ray work={xray_work.item():.4f}, test={xray_test.item():.4f}, Rwork={rwork:.4f}, Rfree={rfree:.4f}")
    
    print("✓ LBFGS test passed")


def quick_test_adam(refinement, target, nsteps=2):
    """Quick Adam test"""
    print(f"\nTesting Adam with target={target}, nsteps={nsteps}")
    refinement.model.freeze_all()
    refinement.model.unfreeze(target)
    refinement.get_scales()
    
    optimizer = torch.optim.Adam(refinement.parameters(), lr=0.01)
    
    for i in range(nsteps):
        optimizer.zero_grad()
        loss = loss_function(refinement)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            xray_work, xray_test = refinement.nll_xray()
            rwork, rfree = refinement.get_rfactor()
            print(f"  Step {i+1}: X-ray work={xray_work.item():.4f}, test={xray_test.item():.4f}, Rwork={rwork:.4f}, Rfree={rfree:.4f}")
    
    print("✓ Adam test passed")


def test_refinement_reload(mtz, pdb, target_weights):
    """Test that refinement can be reloaded correctly"""
    print("\nTesting refinement reload...")
    
    # Create initial refinement
    ref1 = Refinement(mtz, pdb, verbose=0, target_weights=target_weights)
    print(f"  Initial refinement created")
    print(f"    data_file: {ref1.data_file}")
    print(f"    pdb: {ref1.pdb}")
    print(f"    target_weights: {ref1.target_weights}")
    
    # Test reload
    ref2 = Refinement(ref1.data_file, ref1.pdb, verbose=0, target_weights=ref1.target_weights)
    print(f"  Reloaded refinement created")
    print("✓ Reload test passed")
    
    return ref1


if __name__ == "__main__":
    print("="*80)
    print("QUICK OPTIMIZER TEST")
    print("="*80)
    
    # Configuration
    mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
    pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
    target_weights = {'xray': 1.0, 'restraints': 1.0, 'adp': 0.3}
    
    # Test 1: Can we create and reload refinement?
    refinement = test_refinement_reload(mtz, pdb, target_weights)
    
    # Test 2: Quick LBFGS test with xyz
    quick_test_lbfgs(refinement, 'xyz', nsteps=2)
    
    # Test 3: Create fresh refinement and test Adam
    refinement2 = Refinement(mtz, pdb, verbose=0, target_weights=target_weights)
    quick_test_adam(refinement2, 'xyz', nsteps=2)
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✓")
    print("="*80)
    print("\nYou can now run the full benchmark with:")
    print("  sbatch test_optimizers.py")
