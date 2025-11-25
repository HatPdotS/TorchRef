#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
"""
Quick test of the grad_norm fix
"""
import torch
from torchref.base_refinement import Refinement

def grad_norm(loss, params):

    # Zero gradients
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    
    # Compute gradients
    loss.backward(retain_graph=True)
    
    # Collect gradients from parameters that have them
    grad_list = [p.grad.flatten().detach() for p in params if p.grad is not None]
    
    # Check if we have any gradients
    if len(grad_list) == 0:
        # No parameters with gradients - return default value
        print("WARNING: No parameters with gradients found")
        return 1.0
    
    # Concatenate and compute norm
    vec = torch.cat(grad_list)
    norm_val = vec.norm().item()
    print(f"Gradient norm computed: {norm_val:.4f} from {len(grad_list)} parameter tensors")
    return norm_val

# Load data
mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'

print("Creating refinement object...")
refinement = Refinement(pdb=pdb, data_file=mtz, verbose=1)

print("\n" + "="*60)
print("TEST 1: ADP refinement")
print("="*60)
refinement.model.freeze_all()
refinement.model.unfreeze('adp')

print(f"Active parameters: {len(list(refinement.parameters()))}")
for i, p in enumerate(refinement.parameters()):
    print(f"  Param {i}: shape={p.shape}, requires_grad={p.requires_grad}")

loss_xray = refinement.xray_loss()
loss_adp = refinement.adp_loss()

print("\nComputing gradient norms...")
gx = grad_norm(loss_xray, refinement.parameters())
ga = grad_norm(loss_adp, refinement.parameters())

print(f"\nResults:")
print(f"  X-ray gradient norm: {gx:.4f}")
print(f"  ADP gradient norm: {ga:.4f}")
print(f"  Ratio (gx/ga): {gx/ga:.4f}")

print("\n" + "="*60)
print("TEST 2: XYZ refinement")
print("="*60)
refinement.model.freeze_all()
refinement.scaler.freeze()
refinement.model.unfreeze('xyz')

print(f"Active parameters: {len(list(refinement.parameters()))}")
for i, p in enumerate(refinement.parameters()):
    print(f"  Param {i}: shape={p.shape}, requires_grad={p.requires_grad}")

loss_xray = refinement.xray_loss()
loss_geom = refinement.restraints_loss()

print("\nComputing gradient norms...")
gx = grad_norm(loss_xray, refinement.parameters())
gg = grad_norm(loss_geom, refinement.parameters())

print(f"\nResults:")
print(f"  X-ray gradient norm: {gx:.4f}")
print(f"  Geometry gradient norm: {gg:.4f}")
print(f"  Ratio (gx/gg): {gx/gg:.4f}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
