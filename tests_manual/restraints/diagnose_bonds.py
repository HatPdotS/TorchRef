#!/usr/bin/env python3
"""Diagnose bond length NLL issues."""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torchref.model import Model
from torchref.restraints import Restraints

def diagnose_bonds():
    """Diagnose bond length calculation and NLL issues."""
    
    # Load test structure
    pdb_file = Path(__file__).parent.parent.parent / "test_data" / "dark.pdb"
    
    print(f"Loading structure: {pdb_file}")
    model = Model()
    model.load_pdb_from_file(str(pdb_file))
    
    print("\nBuilding restraints...")
    restraints = Restraints(model, verbose=0)
    
    # Get bond data
    if 'all' not in restraints.restraints['bond']:
        restraints.cat_dict()
    
    idx = restraints.restraints['bond']['all']['indices']
    references = restraints.restraints['bond']['all']['references']
    sigmas = restraints.restraints['bond']['all']['sigmas']
    
    # Calculate bond lengths
    calculated = restraints.bond_lengths(idx)
    
    print(f"\n{'='*80}")
    print(f"BOND RESTRAINTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total bonds: {len(idx)}")
    
    # Check for issues
    print(f"\n{'='*80}")
    print(f"DATA QUALITY CHECKS")
    print(f"{'='*80}")
    
    # Check 1: Sigma values
    print(f"\nSigma statistics:")
    print(f"  Mean:   {sigmas.mean().item():.4f} Å")
    print(f"  Median: {sigmas.median().item():.4f} Å")
    print(f"  Min:    {sigmas.min().item():.4f} Å")
    print(f"  Max:    {sigmas.max().item():.4f} Å")
    print(f"  Zeros:  {(sigmas == 0).sum().item()}")
    print(f"  Very small (<0.001): {(sigmas < 0.001).sum().item()}")
    
    # Check 2: Reference values
    print(f"\nReference bond lengths:")
    print(f"  Mean:   {references.mean().item():.4f} Å")
    print(f"  Median: {references.median().item():.4f} Å")
    print(f"  Min:    {references.min().item():.4f} Å")
    print(f"  Max:    {references.max().item():.4f} Å")
    
    # Check 3: Calculated values
    print(f"\nCalculated bond lengths:")
    print(f"  Mean:   {calculated.mean().item():.4f} Å")
    print(f"  Median: {calculated.median().item():.4f} Å")
    print(f"  Min:    {calculated.min().item():.4f} Å")
    print(f"  Max:    {calculated.max().item():.4f} Å")
    
    # Calculate deviations
    print(f"\n{'='*80}")
    print(f"DEVIATION ANALYSIS")
    print(f"{'='*80}")
    
    deviations = calculated - references
    abs_deviations = torch.abs(deviations)
    normalized_deviations = deviations / sigmas
    
    print(f"\nAbsolute deviations:")
    print(f"  Mean:   {abs_deviations.mean().item():.4f} Å")
    print(f"  Median: {abs_deviations.median().item():.4f} Å")
    print(f"  Max:    {abs_deviations.max().item():.4f} Å")
    print(f"  >0.1 Å: {(abs_deviations > 0.1).sum().item()} bonds")
    print(f"  >0.5 Å: {(abs_deviations > 0.5).sum().item()} bonds")
    
    print(f"\nNormalized deviations (dev/sigma):")
    print(f"  Mean:   {normalized_deviations.abs().mean().item():.2f}")
    print(f"  Median: {normalized_deviations.abs().median().item():.2f}")
    print(f"  Max:    {normalized_deviations.abs().max().item():.2f}")
    print(f"  >3σ:    {(normalized_deviations.abs() > 3).sum().item()} bonds")
    print(f"  >10σ:   {(normalized_deviations.abs() > 10).sum().item()} bonds")
    
    # NLL analysis
    print(f"\n{'='*80}")
    print(f"NLL ANALYSIS")
    print(f"{'='*80}")
    
    nll = restraints.nll_bonds()
    
    # Break down NLL components
    squared_term = 0.5 * (deviations / sigmas) ** 2
    log_sigma_term = torch.log(sigmas)
    log_2pi_term = 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
    
    print(f"\nNLL components (mean values):")
    print(f"  0.5*(dev/σ)²:     {squared_term.mean().item():.3f}")
    print(f"  log(σ):           {log_sigma_term.mean().item():.3f}")
    print(f"  0.5*log(2π):      {log_2pi_term.item():.3f}")
    print(f"  {'─'*50}")
    print(f"  Total mean NLL:   {nll.mean().item():.3f}")
    
    print(f"\nNLL statistics:")
    print(f"  Total: {nll.sum().item():.1f}")
    print(f"  Mean:  {nll.mean().item():.3f}")
    print(f"  Median: {nll.median().item():.3f}")
    print(f"  Max:   {nll.max().item():.1f}")
    
    # Expected NLL for perfect fit
    log_sigma_mean = torch.log(sigmas).mean()
    expected_perfect = log_sigma_mean + 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
    print(f"\nExpected NLL (perfect fit, dev=0):")
    print(f"  Mean: {expected_perfect.item():.3f}")
    
    # Find worst bonds
    print(f"\n{'='*80}")
    print(f"TOP 20 WORST BONDS")
    print(f"{'='*80}")
    
    worst_idx = torch.argsort(nll, descending=True)[:20]
    pdb = model.pdb
    
    for i, idx_val in enumerate(worst_idx, 1):
        idx_val = idx_val.item()
        atom_indices = idx[idx_val].cpu().numpy()
        
        # Get atom info
        atom1 = pdb.iloc[atom_indices[0]]
        atom2 = pdb.iloc[atom_indices[1]]
        
        bond_str = f"{atom1['chainid']}:{atom1['resseq']}{atom1['resname']}:{atom1['name']}-{atom2['name']}"
        
        print(f"\n{i:2d}. NLL={nll[idx_val].item():.2f} | {bond_str}")
        print(f"    Calc: {calculated[idx_val].item():.4f} Å | Ref: {references[idx_val].item():.4f} Å | σ: {sigmas[idx_val].item():.4f} Å")
        print(f"    Dev: {deviations[idx_val].item():+.4f} Å ({normalized_deviations[idx_val].item():+.2f}σ)")
    
    # Check if log(sigma) term is dominating
    print(f"\n{'='*80}")
    print(f"DIAGNOSIS")
    print(f"{'='*80}")
    
    if log_sigma_term.mean().item() > 5:
        print(f"\n⚠️  PROBLEM IDENTIFIED: log(σ) term is very large!")
        print(f"   Mean log(σ) = {log_sigma_term.mean().item():.3f}")
        print(f"   This happens when σ values are large (>148 = e^5)")
        print(f"\n   Checking sigma distribution:")
        
        for threshold in [0.1, 0.5, 1.0, 5.0, 10.0, 100.0]:
            count = (sigmas > threshold).sum().item()
            pct = 100 * count / len(sigmas)
            print(f"   σ > {threshold:5.1f} Å: {count:5d} bonds ({pct:5.1f}%)")
    
    if squared_term.mean().item() > 10:
        print(f"\n⚠️  PROBLEM IDENTIFIED: Squared deviation term is very large!")
        print(f"   Mean 0.5*(dev/σ)² = {squared_term.mean().item():.3f}")
        print(f"   This suggests systematic deviations > 3σ")
        
        bad_bonds = (normalized_deviations.abs() > 3).sum().item()
        print(f"   Bonds with |dev| > 3σ: {bad_bonds} ({100*bad_bonds/len(idx):.1f}%)")
    
    # Create plots
    print(f"\n{'='*80}")
    print(f"Creating diagnostic plots...")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Deviation histogram
    ax = axes[0, 0]
    ax.hist(deviations.detach().cpu().numpy(), bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='Perfect')
    ax.set_xlabel('Deviation (Å)')
    ax.set_ylabel('Count')
    ax.set_title('Bond Length Deviations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Normalized deviation histogram
    ax = axes[0, 1]
    ax.hist(normalized_deviations.detach().cpu().numpy(), bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--')
    ax.axvline(-3, color='orange', linestyle='--', alpha=0.5, label='±3σ')
    ax.axvline(3, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('Deviation / σ')
    ax.set_ylabel('Count')
    ax.set_title('Normalized Deviations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: NLL histogram
    ax = axes[0, 2]
    ax.hist(nll.detach().cpu().numpy(), bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(expected_perfect.item(), color='red', linestyle='--', label='Expected (perfect)')
    ax.set_xlabel('NLL')
    ax.set_ylabel('Count')
    ax.set_title('Bond NLL Distribution')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Calculated vs Reference
    ax = axes[1, 0]
    ax.scatter(references.detach().cpu().numpy(), 
              calculated.detach().cpu().numpy(),
              alpha=0.1, s=1)
    ax.plot([0, 3], [0, 3], 'r--', label='Perfect agreement')
    ax.set_xlabel('Reference (Å)')
    ax.set_ylabel('Calculated (Å)')
    ax.set_title('Calculated vs Reference Bond Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 5: Sigma distribution
    ax = axes[1, 1]
    ax.hist(sigmas.detach().cpu().numpy(), bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('σ (Å)')
    ax.set_ylabel('Count')
    ax.set_title('Sigma Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: NLL vs deviation
    ax = axes[1, 2]
    ax.scatter(abs_deviations.detach().cpu().numpy(), 
              nll.detach().cpu().numpy(),
              alpha=0.1, s=1)
    ax.set_xlabel('|Deviation| (Å)')
    ax.set_ylabel('NLL')
    ax.set_title('NLL vs Absolute Deviation')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path(__file__).parent / "bond_diagnostics.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved plot to: {output_file}")
    
    return True

if __name__ == '__main__':
    diagnose_bonds()
