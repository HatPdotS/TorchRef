#!/usr/bin/env python3
"""Diagnose torsion angle NLL issues."""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torchref.model import Model
from torchref.restraints import Restraints

def diagnose_torsions():
    """Diagnose torsion angle calculation and NLL issues."""
    
    # Load test structure
    pdb_file = Path(__file__).parent.parent.parent / "test_data" / "dark.pdb"
    
    print(f"Loading structure: {pdb_file}")
    model = Model()
    model.load_pdb_from_file(str(pdb_file))
    
    print("\nBuilding restraints...")
    restraints = Restraints(model, verbose=0)
    
    # Get torsion data
    if 'all' not in restraints.restraints['torsion']:
        restraints.cat_dict()
    
    idx = restraints.restraints['torsion']['all']['indices']
    references = restraints.restraints['torsion']['all']['references']
    sigmas = restraints.restraints['torsion']['all']['sigmas']
    periods = restraints.restraints['torsion']['all']['periods']
    
    # Calculate torsion angles
    calculated = restraints.torsions(idx)
    
    print(f"\n{'='*80}")
    print(f"TORSION RESTRAINTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total torsions: {len(idx)}")
    print(f"\nPeriods distribution:")
    for period in periods.unique():
        count = (periods == period).sum().item()
        print(f"  Period {period}: {count} torsions ({100*count/len(periods):.1f}%)")
    
    # Calculate deviations
    print(f"\n{'='*80}")
    print(f"DEVIATION ANALYSIS")
    print(f"{'='*80}")
    
    # Raw deviations (without periodicity)
    raw_diff = calculated - references
    print(f"\nRaw deviations (calculated - reference):")
    print(f"  Mean: {raw_diff.mean():.2f}°")
    print(f"  Std:  {raw_diff.std():.2f}°")
    print(f"  Min:  {raw_diff.min():.2f}°")
    print(f"  Max:  {raw_diff.max():.2f}°")
    
    # Properly wrapped deviations
    wrapped_dev = restraints.torsion_deviations(wrapped=True)
    print(f"\nWrapped deviations (accounting for periodicity):")
    print(f"  Mean: {wrapped_dev.mean():.2f}°")
    print(f"  Std:  {wrapped_dev.std():.2f}°")
    print(f"  Min:  {wrapped_dev.min():.2f}°")
    print(f"  Max:  {wrapped_dev.max():.2f}°")
    
    # Check the periodicity application
    print(f"\n{'='*80}")
    print(f"PERIODICITY APPLICATION ISSUE")
    print(f"{'='*80}")
    
    # Current method (in nll_torsions)
    torsion_rad = calculated * torch.pi / 180.0
    expectation_rad = references * torch.pi / 180.0
    
    # THIS IS THE PROBLEM: multiplying by period BEFORE taking difference
    diff_current = periods * (torsion_rad - expectation_rad)
    diff_current_wrapped = torch.atan2(torch.sin(diff_current), torch.cos(diff_current))
    
    # CORRECT METHOD: take difference first, then handle periodicity
    diff_correct = torsion_rad - expectation_rad
    diff_correct_wrapped = torch.atan2(torch.sin(diff_correct), torch.cos(diff_correct))
    diff_correct_adjusted = diff_correct_wrapped / periods.float()
    
    print(f"\nCurrent method (period * (calc - ref)):")
    print(f"  Mean wrapped diff: {torch.rad2deg(diff_current_wrapped).mean():.2f}°")
    print(f"  Std wrapped diff:  {torch.rad2deg(diff_current_wrapped).std():.2f}°")
    print(f"  Min/Max: [{torch.rad2deg(diff_current_wrapped).min():.2f}, {torch.rad2deg(diff_current_wrapped).max():.2f}]°")
    
    print(f"\nCorrect method ((calc - ref) / period):")
    print(f"  Mean adjusted diff: {torch.rad2deg(diff_correct_adjusted).mean():.2f}°")
    print(f"  Std adjusted diff:  {torch.rad2deg(diff_correct_adjusted).std():.2f}°")
    print(f"  Min/Max: [{torch.rad2deg(diff_correct_adjusted).min():.2f}, {torch.rad2deg(diff_correct_adjusted).max():.2f}]°")
    
    # NLL comparison
    print(f"\n{'='*80}")
    print(f"NLL ANALYSIS")
    print(f"{'='*80}")
    
    # Current NLL
    nll_current = restraints.nll_torsions()
    print(f"\nCurrent NLL calculation:")
    print(f"  Total NLL: {nll_current.sum().item():.1f}")
    print(f"  Mean NLL:  {nll_current.mean().item():.3f}")
    print(f"  Median NLL: {nll_current.median().item():.3f}")
    print(f"  Max NLL:   {nll_current.max().item():.1f}")
    
    # Expected NLL for perfect agreement (just normalization constant)
    kappa = torch.clamp(1.0 / (sigmas.deg2rad()**2), min=1e-3, max=1e4)
    from scipy.special import i0
    log_i0_kappa = torch.log(torch.tensor([i0(k.item()) for k in kappa]))
    log_2pi = np.log(2.0 * np.pi)
    expected_nll_perfect = log_i0_kappa + log_2pi
    
    print(f"\nExpected NLL (perfect fit, just normalization):")
    print(f"  Total: {expected_nll_perfect.sum().item():.1f}")
    print(f"  Mean:  {expected_nll_perfect.mean().item():.3f}")
    
    # Show examples with large NLL
    print(f"\n{'='*80}")
    print(f"TOP 10 WORST TORSIONS")
    print(f"{'='*80}")
    
    worst_idx = torch.argsort(nll_current, descending=True)[:10]
    pdb = model.pdb
    
    for i, idx_val in enumerate(worst_idx, 1):
        idx_val = idx_val.item()
        atom_indices = idx[idx_val].cpu().numpy()
        
        # Get atom info
        atoms = [pdb.iloc[ai] for ai in atom_indices]
        atom_str = ' - '.join([f"{a['chainid']}:{a['resseq']}{a['resname']}:{a['name']}" for a in atoms])
        
        print(f"\n{i}. NLL={nll_current[idx_val].item():.2f}")
        print(f"   Atoms: {atom_str}")
        print(f"   Calculated: {calculated[idx_val].item():.1f}°")
        print(f"   Reference:  {references[idx_val].item():.1f}°")
        print(f"   Sigma:      {sigmas[idx_val].item():.1f}°")
        print(f"   Period:     {periods[idx_val].item()}")
        print(f"   Deviation:  {wrapped_dev[idx_val].item():.1f}°")
        print(f"   Dev/Sigma:  {(wrapped_dev[idx_val] / sigmas[idx_val]).item():.2f}")
    
    # Create diagnostic plot
    print(f"\n{'='*80}")
    print(f"Creating diagnostic plots...")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Deviation distribution
    ax = axes[0, 0]
    ax.hist(wrapped_dev.detach().cpu().numpy(), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='Zero deviation')
    ax.set_xlabel('Deviation (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Torsion Deviation Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: NLL distribution
    ax = axes[0, 1]
    ax.hist(nll_current.detach().cpu().numpy(), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(expected_nll_perfect.mean().item(), color='red', linestyle='--', 
               label='Expected (perfect fit)')
    ax.set_xlabel('NLL')
    ax.set_ylabel('Count')
    ax.set_title('Torsion NLL Distribution')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Deviation vs Period
    ax = axes[1, 0]
    for period in periods.unique():
        mask = periods == period
        ax.scatter(wrapped_dev[mask].detach().cpu().numpy(), 
                  nll_current[mask].detach().cpu().numpy(),
                  alpha=0.3, s=10, label=f'Period {period.item()}')
    ax.set_xlabel('Deviation (degrees)')
    ax.set_ylabel('NLL')
    ax.set_title('NLL vs Deviation by Period')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Calculated vs Reference
    ax = axes[1, 1]
    ax.scatter(references.detach().cpu().numpy(), 
              calculated.detach().cpu().numpy(),
              alpha=0.1, s=5)
    ax.plot([-180, 180], [-180, 180], 'r--', label='Perfect agreement')
    ax.set_xlabel('Reference angle (degrees)')
    ax.set_ylabel('Calculated angle (degrees)')
    ax.set_title('Calculated vs Reference Angles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    
    plt.tight_layout()
    
    output_file = Path(__file__).parent / "torsion_diagnostics.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved plot to: {output_file}")
    
    return True

if __name__ == '__main__':
    diagnose_torsions()
