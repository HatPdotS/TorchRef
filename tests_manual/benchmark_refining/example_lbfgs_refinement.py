#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/test_lbfgs_refinement.out

"""
Example script demonstrating the LBFGSRefinement class.

This shows how to use the new LBFGS-based refinement framework which 
converges much faster than traditional Adam-based refinement.
"""

import torch
from torchref.lbfgs_refinement import LBFGSRefinement, LBFGS_scales
from torchref.base_refinement import Refinement


def example_basic_lbfgs():
    """Basic example: Run LBFGS refinement with default settings"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic LBFGS Refinement")
    print("="*80 + "\n")
    
    # Data files
    mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
    pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
    
    # Create LBFGSRefinement object
    refinement = LBFGSRefinement(
        mtz, pdb,
        verbose=1,
        target_weights={'xray': 1.0, 'restraints': 1.0, 'adp': 0.3}
    )
    
    # Run refinement - typically converges in just 2 macro cycles!
    metrics = refinement.run_lbfgs_refinement(
        macro_cycles=2,
        targets=['xyz', 'b'],
        steps_per_target=3,
        steps_scales=5
    )
    
    # Get final summary
    summary = refinement.get_refinement_summary()
    print("\nFinal Summary:")
    print(f"  R-work: {summary['rwork']:.4f}")
    print(f"  R-free: {summary['rfree']:.4f}")
    print(f"  Test/Work ratio: {summary['ratio_test_work']:.3f}")
    
    return refinement, metrics


def example_advanced_lbfgs():
    """Advanced example: Customize LBFGS parameters"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Advanced LBFGS Refinement with Custom Parameters")
    print("="*80 + "\n")
    
    mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
    pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
    
    refinement = LBFGSRefinement(
        mtz, pdb,
        verbose=2,
        target_weights={'xray': 1.0, 'restraints': 0.8, 'adp': 0.3}
    )
    
    # Run with custom LBFGS parameters
    metrics = refinement.run_lbfgs_refinement(
        macro_cycles=3,
        targets=['xyz', 'b', 'occ'],  # Also refine occupancies
        steps_per_target=5,  # More steps per target
        steps_scales=10,  # More steps for scales
        lr=1.0,
        max_iter=25,  # More line search iterations
        history_size=15,  # Larger history for better Hessian approximation
        line_search_fn='strong_wolfe',  # Use strong Wolfe line search
        refine_scales=True,
        update_weights=True,
        outlier_rejection=True,
        z_threshold=4.0
    )
    
    # Write out refined structure
    refinement.model.write_pdb('refined_lbfgs.pdb')
    refinement.write_out_mtz('refined_lbfgs.mtz')
    
    print("\nOutput files written:")
    print("  refined_lbfgs.pdb")
    print("  refined_lbfgs.mtz")
    
    return refinement, metrics


def example_scales_only():
    """Example: Refine only scales using LBFGS"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Scale Refinement Only")
    print("="*80 + "\n")
    
    mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
    pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
    
    # Can use regular Refinement class
    refinement = Refinement(
        mtz, pdb,
        verbose=1,
        target_weights={'xray': 1.0, 'restraints': 1.0, 'adp': 0.3}
    )
    
    # Initial scale fitting
    refinement.get_scales()
    
    # Then refine with LBFGS using standalone function
    metrics = LBFGS_scales(refinement, nsteps=10, verbose=True)
    
    print(f"\nFinal R-factors after scale refinement:")
    rwork, rfree = refinement.get_rfactor()
    print(f"  R-work: {rwork:.4f}")
    print(f"  R-free: {rfree:.4f}")
    
    return refinement, metrics


def example_stepwise_refinement():
    """Example: Step-by-step refinement with manual control"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Manual Step-by-Step Refinement")
    print("="*80 + "\n")
    
    mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
    pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
    
    refinement = LBFGSRefinement(
        mtz, pdb,
        verbose=1,
        target_weights={'xray': 1.0, 'restraints': 1.0, 'adp': 0.3}
    )
    
    all_metrics = []
    
    # Step 1: Initialize scales
    print("\nStep 1: Initializing scales...")
    refinement.get_scales()
    
    # Step 2: Refine scales with LBFGS
    print("\nStep 2: Refining scales...")
    metrics_scales = refinement.refine_scales_lbfgs(nsteps=5)
    all_metrics.append(metrics_scales)
    
    # Step 3: Refine coordinates
    print("\nStep 3: Refining coordinates...")
    metrics_xyz = refinement.refine_target_lbfgs('xyz', nsteps=3)
    all_metrics.append(metrics_xyz)
    
    # Step 4: Refine B-factors
    print("\nStep 4: Refining B-factors...")
    metrics_b = refinement.refine_target_lbfgs('b', nsteps=3)
    all_metrics.append(metrics_b)
    
    # Step 5: One more round of coordinates
    print("\nStep 5: Second round of coordinates...")
    metrics_xyz2 = refinement.refine_target_lbfgs('xyz', nsteps=3)
    all_metrics.append(metrics_xyz2)
    
    # Print progression
    print("\n" + "="*80)
    print("REFINEMENT PROGRESSION")
    print("="*80)
    for i, m in enumerate(all_metrics):
        target = m['target']
        initial_rfree = m['rfree'][0]
        final_rfree = m['rfree'][-1]
        improvement = initial_rfree - final_rfree
        print(f"{i+1}. {target:8s}: Rfree {initial_rfree:.4f} → {final_rfree:.4f} (Δ={improvement:+.4f})")
    
    return refinement, all_metrics


def compare_adam_vs_lbfgs():
    """Example: Compare Adam vs LBFGS refinement"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Compare Adam vs LBFGS")
    print("="*80 + "\n")
    
    mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
    pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
    
    # Test with Adam (base class)
    print("Running Adam-based refinement (5 macro cycles)...")
    refinement_adam = Refinement(
        mtz, pdb,
        verbose=0,
        target_weights={'xray': 1.0, 'restraints': 1.0, 'adp': 0.3}
    )
    import time
    start = time.time()
    refinement_adam.run_refinement(macro_cycles=5, n_steps=10)
    adam_time = time.time() - start
    rwork_adam, rfree_adam = refinement_adam.get_rfactor()
    
    # Test with LBFGS
    print("\nRunning LBFGS-based refinement (2 macro cycles)...")
    refinement_lbfgs = LBFGSRefinement(
        mtz, pdb,
        verbose=0,
        target_weights={'xray': 1.0, 'restraints': 1.0, 'adp': 0.3}
    )
    start = time.time()
    refinement_lbfgs.run_lbfgs_refinement(macro_cycles=2, steps_per_target=3)
    lbfgs_time = time.time() - start
    rwork_lbfgs, rfree_lbfgs = refinement_lbfgs.get_rfactor()
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"\n{'Method':<15} {'Cycles':<10} {'Time (s)':<12} {'R-work':<10} {'R-free':<10}")
    print("-"*80)
    print(f"{'Adam':<15} {5:<10} {adam_time:<12.2f} {rwork_adam:<10.4f} {rfree_adam:<10.4f}")
    print(f"{'LBFGS':<15} {2:<10} {lbfgs_time:<12.2f} {rwork_lbfgs:<10.4f} {rfree_lbfgs:<10.4f}")
    print("-"*80)
    print(f"\nLBFGS is {adam_time/lbfgs_time:.1f}x faster")
    print(f"LBFGS R-free improvement: {rfree_adam - rfree_lbfgs:+.4f}")
    
    return refinement_adam, refinement_lbfgs


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# LBFGS REFINEMENT EXAMPLES")
    print("#"*80)
    
    # Choose which examples to run
    RUN_BASIC = True
    RUN_ADVANCED = False
    RUN_SCALES_ONLY = False
    RUN_STEPWISE = False
    RUN_COMPARISON = False
    
    if RUN_BASIC:
        refinement1, metrics1 = example_basic_lbfgs()
    
    if RUN_ADVANCED:
        refinement2, metrics2 = example_advanced_lbfgs()
    
    if RUN_SCALES_ONLY:
        refinement3, metrics3 = example_scales_only()
    
    if RUN_STEPWISE:
        refinement4, metrics4 = example_stepwise_refinement()
    
    if RUN_COMPARISON:
        ref_adam, ref_lbfgs = compare_adam_vs_lbfgs()
    
    print("\n" + "#"*80)
    print("# ALL EXAMPLES COMPLETE")
    print("#"*80 + "\n")
