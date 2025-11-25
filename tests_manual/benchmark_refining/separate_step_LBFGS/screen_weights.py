#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/separate_step_LBFGS/weight_screening.out
#SBATCH -p week
#SBATCH -t 2-00:00:00 
import torch
import numpy as np
from torchref.base_refinement import Refinement
import json
import os
from datetime import datetime


def refine(refinement, weights={'xray':1.0, 'restraints':10.0, 'adp':0.3}, n_cycles=10):
    """Run refinement for n_cycles alternating between ADP and XYZ"""
    refinement.effective_weights = weights
    refinement.scaler.freeze()
    
    for cycle in range(n_cycles):
        refinement.get_scales()

        refine_xyz(refinement, weights=weights)
        refine_adp(refinement, weights=weights)

    
    return refinement.model, refinement.get_rfactor()[1]


def grad_norm(loss, refinement):
    """Compute gradient norm with proper zeroing"""

    # Zero gradients
    for p in refinement.model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    
    # Compute gradients
    loss.backward(retain_graph=True)
    
    # Collect gradients from parameters that have them
    grad_list = [p.grad.flatten().detach() for p in refinement.model.parameters() if p.grad is not None]
    
    # Check if we have any gradients
    if len(grad_list) == 0:
        # No parameters with gradients - return default value
        return 1.0
    
    # Concatenate and compute norm
    vec = torch.cat(grad_list)
    return vec.norm().item()



def refine_adp(refinement, weights={'xray':1.0, 'restraints':1.0, 'adp':0.3}):
    """Refine B-factors (ADP)"""
    refinement.model.freeze_all()
    refinement.model.unfreeze('b')
    
    # Compute gradient-based weight using refinement.parameters() not model.parameters()
    loss_xray_ = refinement.xray_loss()
    loss_adp_ = refinement.adp_loss()

    gx = grad_norm(loss_xray_, refinement)
    ga = grad_norm(loss_adp_, refinement)
    weight_adp = (gx / (ga + 1e-12)) * weights['adp']
    refinement.target_weights['adp'] = weight_adp

    optimizer = torch.optim.LBFGS(
        refinement.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = loss_adp(refinement)
        loss.backward(retain_graph=True)
        return loss

    optimizer.step(closure)
    refinement.model.unfreeze_all()


def refine_xyz(refinement, weights={'xray':1.0, 'restraints':10.0, 'adp':0.3}):
    """Refine coordinates (XYZ)"""
    refinement.model.freeze_all()
    refinement.scaler.freeze()
    refinement.model.unfreeze('xyz')

    # Compute gradient-based weight
    loss_xray_ = refinement.xray_loss()
    loss_geom_ = refinement.restraints_loss()

    gx = grad_norm(loss_xray_, refinement)
    gg = grad_norm(loss_geom_, refinement)

    weight_restraints = (gx / (gg + 1e-12)) * weights['restraints']
    refinement.target_weights['restraints'] = weight_restraints

    optimizer = torch.optim.LBFGS(
        refinement.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = loss_geom(refinement)
        loss.backward(retain_graph=True)
        return loss

    optimizer.step(closure)
    refinement.model.unfreeze_all()


def loss_geom(refinement):
    """Loss for XYZ refinement"""
    loss = 0.0
    if 'xray' in refinement.target_weights:
        loss = loss + refinement.target_weights['xray'] * refinement.xray_loss()    
    if 'restraints' in refinement.target_weights:
        loss = loss + refinement.target_weights['restraints'] * refinement.restraints_loss()
    return loss


def loss_adp(refinement):
    """Loss for ADP refinement"""
    loss = 0.0
    if 'xray' in refinement.target_weights:
        loss = loss + refinement.target_weights['xray'] * refinement.xray_loss()
    if 'adp' in refinement.target_weights:
        loss = loss + refinement.target_weights['adp'] * refinement.adp_loss()
    return loss


def screen_weights(mtz, pdb, outdir, n_cycles=10):
    """Screen different weight combinations"""
    
    # Generate log-scale weight values
    # 5 steps from 0.2 to 2.0 on log scale
    restraints_weights = np.logspace(np.log10(0.2), np.log10(2.0), 5)
    adp_weights = np.logspace(np.log10(0.2), np.log10(2.0), 5)
    
    print("=" * 80)
    print("WEIGHT SCREENING")
    print("=" * 80)
    print(f"Restraints weights: {restraints_weights}")
    print(f"ADP weights: {adp_weights}")
    print(f"Number of cycles per run: {n_cycles}")
    print(f"Total combinations: {len(restraints_weights) * len(adp_weights)}")
    print("=" * 80)
    
    results = []
    
    # Screen all weight combinations
    for i, restraints_wt in enumerate(restraints_weights):
        for j, adp_wt in enumerate(adp_weights):
            
            combo_id = i * len(adp_weights) + j + 1
            total_combos = len(restraints_weights) * len(adp_weights)
            
            print("\n" + "=" * 80)
            print(f"Combination {combo_id}/{total_combos}")
            print(f"Restraints weight: {restraints_wt:.4f}, ADP weight: {adp_wt:.4f}")
            print("=" * 80)
            
            # Create fresh refinement object for each combination
            refinement = Refinement(
                pdb=pdb,
                data_file=mtz,
                verbose=1,  # Less verbose for screening
            )
            
            # Get initial R-factors
            with torch.no_grad():
                rwork_init, rfree_init = refinement.get_rfactor()
            
            print(f"Initial: Rwork={rwork_init:.4f}, Rfree={rfree_init:.4f}")
            
            # Run refinement
            weights = {
                'xray': 1.0,
                'restraints': restraints_wt,
                'adp': adp_wt
            }
            

            best_model, best_rfree = refine(refinement, weights=weights, n_cycles=n_cycles)
            
            # Get final R-factors
            with torch.no_grad():
                rwork_final, rfree_final = refinement.get_rfactor()
            
            print(f"Final: Rwork={rwork_final:.4f}, Rfree={rfree_final:.4f}")
            print(f"Delta Rfree: {rfree_final - rfree_init:+.4f}")
            
            # Save results
            result = {
                'combo_id': combo_id,
                'restraints_weight': float(restraints_wt),
                'adp_weight': float(adp_wt),
                'rwork_initial': float(rwork_init),
                'rfree_initial': float(rfree_init),
                'rwork_final': float(rwork_final),
                'rfree_final': float(rfree_final),
                'delta_rwork': float(rwork_final - rwork_init),
                'delta_rfree': float(rfree_final - rfree_init),
                'n_cycles': n_cycles,
                'status': 'success'
            }
            
            # Save model
            model_file = f"{outdir}/screened_model_r{restraints_wt:.3f}_a{adp_wt:.3f}.pdb"
            best_model.write_pdb(model_file)
            result['model_file'] = model_file
                
            # except Exception as e:
            #     print(f"ERROR: Refinement failed with exception: {e}")
            #     result = {
            #         'combo_id': combo_id,
            #         'restraints_weight': float(restraints_wt),
            #         'adp_weight': float(adp_wt),
            #         'rwork_initial': float(rwork_init),
            #         'rfree_initial': float(rfree_init),
            #         'rwork_final': None,
            #         'rfree_final': None,
            #         'delta_rwork': None,
            #         'delta_rfree': None,
            #         'n_cycles': n_cycles,
            #         'status': 'failed',
            #         'error': str(e)
            #     }
            
            results.append(result)
            
            # Clean up memory before next iteration
            del refinement
            del best_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            # Save intermediate results after each combination
            save_results(results, outdir)
    
    # Print summary
    print_summary(results)
    
    return results


def save_results(results, outdir):
    """Save results to JSON file"""
    output_file = f"{outdir}/weight_screening_results.json"
    
    # Add metadata
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'n_combinations': len(results),
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def print_summary(results):
    """Print summary of screening results"""
    print("\n" + "=" * 80)
    print("SCREENING SUMMARY")
    print("=" * 80)
    
    # Filter successful results
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        print("No successful refinements!")
        return
    
    # Sort by final Rfree
    successful_sorted = sorted(successful, key=lambda x: x['rfree_final'])
    
    print(f"\nTotal combinations: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    
    print("\n" + "-" * 80)
    print("TOP 5 RESULTS (by final Rfree):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Restraints':<12} {'ADP':<12} {'Rfree Final':<14} {'Delta Rfree':<12} {'Rwork Final':<12}")
    print("-" * 80)
    
    for rank, result in enumerate(successful_sorted[:5], 1):
        print(f"{rank:<6} {result['restraints_weight']:<12.4f} {result['adp_weight']:<12.4f} "
              f"{result['rfree_final']:<14.4f} {result['delta_rfree']:<+12.4f} "
              f"{result['rwork_final']:<12.4f}")
    
    # Best improvement
    best_improvement = min(successful, key=lambda x: x['delta_rfree'])
    print("\n" + "-" * 80)
    print("BEST IMPROVEMENT:")
    print("-" * 80)
    print(f"Restraints weight: {best_improvement['restraints_weight']:.4f}")
    print(f"ADP weight: {best_improvement['adp_weight']:.4f}")
    print(f"Initial Rfree: {best_improvement['rfree_initial']:.4f}")
    print(f"Final Rfree: {best_improvement['rfree_final']:.4f}")
    print(f"Improvement: {best_improvement['delta_rfree']:.4f}")
    
    # Best final Rfree
    best_final = successful_sorted[0]
    print("\n" + "-" * 80)
    print("BEST FINAL RFREE:")
    print("-" * 80)
    print(f"Restraints weight: {best_final['restraints_weight']:.4f}")
    print(f"ADP weight: {best_final['adp_weight']:.4f}")
    print(f"Initial Rfree: {best_final['rfree_initial']:.4f}")
    print(f"Final Rfree: {best_final['rfree_final']:.4f}")
    print(f"Delta Rfree: {best_final['delta_rfree']:+.4f}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Input files
    mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
    pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
    outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/separate_step_LBFGS/screening_results'
    
    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    
    # Run screening with 10 cycles per combination
    results = screen_weights(mtz, pdb, outdir, n_cycles=10)
    
    print("\nWeight screening complete!")
