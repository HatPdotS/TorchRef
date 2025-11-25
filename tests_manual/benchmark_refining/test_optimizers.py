#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/test_optimizers.out

import torch
import time
from tqdm import tqdm
from torchref.base_refinement import Refinement


# Shared loss function - ensures all optimizers use the same loss calculation
def loss_function(refinement):
    loss = 0.0
    if 'xray' in refinement.target_weights:
        loss = loss + refinement.target_weights['xray'] * refinement.xray_loss()
    if 'restraints' in refinement.target_weights:
        loss = loss + refinement.target_weights['restraints'] * refinement.restraints_loss()
    if 'adp' in refinement.target_weights:
        loss = loss + refinement.target_weights['adp'] * refinement.adp_loss()
    return loss


def refine_lbfgs(refinement, target, nsteps, lr=1.0):
    """LBFGS optimizer - requires closure function"""
    refinement.model.freeze_all()
    refinement.model.unfreeze(target)
    refinement.get_scales()
    
    optimizer = torch.optim.LBFGS(refinement.parameters(), lr=lr, max_iter=20, history_size=10)
    
    def closure():
        optimizer.zero_grad()
        loss = loss_function(refinement)
        loss.backward(retain_graph=True)
        return loss
    
    metrics = []
    start_time = time.time()
    
    for i in range(nsteps):
        optimizer.step(closure)
        with torch.no_grad():
            xray_work, xray_test = refinement.nll_xray()
            rwork, rfree = refinement.get_rfactor()
            metrics.append({
                'step': i+1,
                'xray_work': xray_work.item(),
                'xray_test': xray_test.item(),
                'rwork': rwork,
                'rfree': rfree
            })
    
    elapsed = time.time() - start_time
    return metrics, elapsed


def refine_adam(refinement, target, nsteps, lr=0.01):
    """Adam optimizer"""
    refinement.model.freeze_all()
    refinement.model.unfreeze(target)
    refinement.get_scales()
    
    optimizer = torch.optim.Adam(refinement.parameters(), lr=lr)
    
    metrics = []
    start_time = time.time()
    
    for i in range(nsteps):
        optimizer.zero_grad()
        loss = loss_function(refinement)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            xray_work, xray_test = refinement.nll_xray()
            rwork, rfree = refinement.get_rfactor()
            metrics.append({
                'step': i+1,
                'xray_work': xray_work.item(),
                'xray_test': xray_test.item(),
                'rwork': rwork,
                'rfree': rfree
            })
    
    elapsed = time.time() - start_time
    return metrics, elapsed


def refine_adamw(refinement, target, nsteps, lr=0.01):
    """AdamW optimizer - Adam with decoupled weight decay"""
    refinement.model.freeze_all()
    refinement.model.unfreeze(target)
    refinement.get_scales()
    
    optimizer = torch.optim.AdamW(refinement.parameters(), lr=lr, weight_decay=0.01)
    
    metrics = []
    start_time = time.time()
    
    for i in range(nsteps):
        optimizer.zero_grad()
        loss = loss_function(refinement)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            xray_work, xray_test = refinement.nll_xray()
            rwork, rfree = refinement.get_rfactor()
            metrics.append({
                'step': i+1,
                'xray_work': xray_work.item(),
                'xray_test': xray_test.item(),
                'rwork': rwork,
                'rfree': rfree
            })
    
    elapsed = time.time() - start_time
    return metrics, elapsed


def refine_sgd(refinement, target, nsteps, lr=0.01):
    """SGD optimizer with momentum"""
    refinement.model.freeze_all()
    refinement.model.unfreeze(target)
    refinement.get_scales()
    
    optimizer = torch.optim.SGD(refinement.parameters(), lr=lr, momentum=0.9)
    
    metrics = []
    start_time = time.time()
    
    for i in range(nsteps):
        optimizer.zero_grad()
        loss = loss_function(refinement)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            xray_work, xray_test = refinement.nll_xray()
            rwork, rfree = refinement.get_rfactor()
            metrics.append({
                'step': i+1,
                'xray_work': xray_work.item(),
                'xray_test': xray_test.item(),
                'rwork': rwork,
                'rfree': rfree
            })
    
    elapsed = time.time() - start_time
    return metrics, elapsed


def refine_rmsprop(refinement, target, nsteps, lr=0.01):
    """RMSprop optimizer"""
    refinement.model.freeze_all()
    refinement.model.unfreeze(target)
    refinement.get_scales()
    
    optimizer = torch.optim.RMSprop(refinement.parameters(), lr=lr, alpha=0.99)
    
    metrics = []
    start_time = time.time()
    
    for i in range(nsteps):
        optimizer.zero_grad()
        loss = loss_function(refinement)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            xray_work, xray_test = refinement.nll_xray()
            rwork, rfree = refinement.get_rfactor()
            metrics.append({
                'step': i+1,
                'xray_work': xray_work.item(),
                'xray_test': xray_test.item(),
                'rwork': rwork,
                'rfree': rfree
            })
    
    elapsed = time.time() - start_time
    return metrics, elapsed


def refine_adagrad(refinement, target, nsteps, lr=0.01):
    """Adagrad optimizer"""
    refinement.model.freeze_all()
    refinement.model.unfreeze(target)
    refinement.get_scales()
    
    optimizer = torch.optim.Adagrad(refinement.parameters(), lr=lr)
    
    metrics = []
    start_time = time.time()
    
    for i in range(nsteps):
        optimizer.zero_grad()
        loss = loss_function(refinement)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            xray_work, xray_test = refinement.nll_xray()
            rwork, rfree = refinement.get_rfactor()
            metrics.append({
                'step': i+1,
                'xray_work': xray_work.item(),
                'xray_test': xray_test.item(),
                'rwork': rwork,
                'rfree': rfree
            })
    
    elapsed = time.time() - start_time
    return metrics, elapsed


def print_results(optimizer_name, target, metrics, elapsed):
    """Print formatted results for an optimizer"""
    print(f"\n{'='*80}")
    print(f"Optimizer: {optimizer_name} | Target: {target}")
    print(f"{'='*80}")
    print(f"Total time: {elapsed:.2f}s | Avg time/step: {elapsed/len(metrics):.3f}s")
    print(f"\n{'Step':<6} {'X-ray Work':<12} {'X-ray Test':<12} {'R-work':<10} {'R-free':<10}")
    print(f"{'-'*60}")
    for m in metrics:
        print(f"{m['step']:<6} {m['xray_work']:<12.4f} {m['xray_test']:<12.4f} {m['rwork']:<10.4f} {m['rfree']:<10.4f}")
    
    # Print final improvements
    if len(metrics) > 0:
        initial = metrics[0]
        final = metrics[-1]
        print(f"\n{'Improvement:':<20}")
        print(f"  X-ray Work: {initial['xray_work']:.4f} -> {final['xray_work']:.4f} (Δ={final['xray_work']-initial['xray_work']:.4f})")
        print(f"  X-ray Test: {initial['xray_test']:.4f} -> {final['xray_test']:.4f} (Δ={final['xray_test']-initial['xray_test']:.4f})")
        print(f"  R-work:     {initial['rwork']:.4f} -> {final['rwork']:.4f} (Δ={final['rwork']-initial['rwork']:.4f})")
        print(f"  R-free:     {initial['rfree']:.4f} -> {final['rfree']:.4f} (Δ={final['rfree']-initial['rfree']:.4f})")


def compare_optimizers(refinement, target, nsteps, optimizers_config):
    """
    Compare multiple optimizers on the same refinement task
    
    Args:
        refinement: Refinement object (will be reset between optimizers)
        target: Target parameter to refine ('xyz', 'b', etc.)
        nsteps: Number of optimization steps
        optimizers_config: Dict of {name: (refine_func, lr)}
    """
    results = {}
    
    for opt_name, (refine_func, lr) in optimizers_config.items():
        print(f"\n\nTesting {opt_name}...")
        
        # Reload refinement to ensure fair comparison
        refinement_copy = Refinement(
            refinement.data_file,
            refinement.pdb,
            verbose=0,
            target_weights=refinement.target_weights
        )
        
        metrics, elapsed = refine_func(refinement_copy, target, nsteps, lr=lr)
        results[opt_name] = {'metrics': metrics, 'elapsed': elapsed}
        print_results(opt_name, target, metrics, elapsed)
    
    return results


def print_summary(all_results):
    """Print a summary comparison of all optimizers"""
    print(f"\n\n{'='*80}")
    print(f"SUMMARY: Optimizer Comparison")
    print(f"{'='*80}\n")
    
    for target, results in all_results.items():
        print(f"\nTarget: {target}")
        print(f"{'-'*80}")
        print(f"{'Optimizer':<15} {'Final R-work':<15} {'Final R-free':<15} {'Time (s)':<12}")
        print(f"{'-'*80}")
        
        for opt_name, data in results.items():
            metrics = data['metrics']
            elapsed = data['elapsed']
            if len(metrics) > 0:
                final = metrics[-1]
                print(f"{opt_name:<15} {final['rwork']:<15.4f} {final['rfree']:<15.4f} {elapsed:<12.2f}")


if __name__ == "__main__":
    # Configuration
    mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
    pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
    
    # Consistent target weights across all tests
    target_weights = {'xray': 1.0, 'restraints': 1.0, 'adp': 0.3}
    
    # Initialize refinement
    print("Initializing refinement...")
    refinement = Refinement(mtz, pdb, verbose=0, target_weights=target_weights)
    
    # Define optimizers to test with their learning rates
    optimizers_config = {
        'LBFGS': (refine_lbfgs, 1.0),
        'Adam': (refine_adam, 0.01),
        'AdamW': (refine_adamw, 0.01),
        'SGD': (refine_sgd, 0.01),
        'RMSprop': (refine_rmsprop, 0.01),
        'Adagrad': (refine_adagrad, 0.01),
    }
    
    # Targets to refine
    targets_to_test = ['xyz', 'b']
    
    # Number of steps per optimizer
    nsteps = 10
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZER BENCHMARK")
    print(f"{'='*80}")
    print(f"Data: {mtz}")
    print(f"Model: {pdb}")
    print(f"Target weights: {target_weights}")
    print(f"Optimizers: {', '.join(optimizers_config.keys())}")
    print(f"Targets: {', '.join(targets_to_test)}")
    print(f"Steps per optimizer: {nsteps}")
    print(f"{'='*80}\n")
    
    # Run comparison for each target
    all_results = {}
    for target in targets_to_test:
        print(f"\n\n{'#'*80}")
        print(f"# Testing target: {target}")
        print(f"{'#'*80}")
        all_results[target] = compare_optimizers(refinement, target, nsteps, optimizers_config)
    
    # Print final summary
    print_summary(all_results)
