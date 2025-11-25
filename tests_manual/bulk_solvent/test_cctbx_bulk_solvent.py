#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBACTH -c 32
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/test_cctbx_bulk_solvent.out
#SBATCH --mem=300G
from mmtbx.bulk_solvent import bulk_solvent_and_scaling
from torchref.model_ft import ModelFT
from torchref.solvent import SolventModel

import iotbx.pdb
import numpy as np

pdb_file_P21 = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/dark.pdb'
pdb_file_P1 = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/dark_P1.pdb'


def test_for_pdb_file(pdb_file,prefix='',dilation_radius=1.3):
    pdb_inp = iotbx.pdb.input(file_name=pdb_file)
    pdb_hierarchy = pdb_inp.construct_hierarchy()
    xray_structure = pdb_inp.xray_structure_simple()
    print(xray_structure)

    from mmtbx.masks import mask_master_params
    from mmtbx import masks


    mask_params = mask_master_params.extract()
    mask_params.solvent_radius = 1.1  # Typical probe radius
    mask_params.shrink_truncation_radius = 0.9  # Shrink radius
    d_min = 1.7

    # IMPORTANT: For non-P1 space groups, we need to expand the structure to P1
    # first to include all symmetry-related copies, otherwise the mask will only
    # cover the asymmetric unit and mark symmetry mates as solvent!
    
    # Option 1: Expand structure to P1 before masking (RECOMMENDED)
    print(f"Original space group: {xray_structure.space_group_info()}")
    print(f"Original n_atoms: {xray_structure.scatterers().size()}")
    
    # Expand to P1 symmetry
    xray_structure_p1 = xray_structure.expand_to_p1()
    print(f"Expanded space group: {xray_structure_p1.space_group_info()}")
    print(f"Expanded n_atoms: {xray_structure_p1.scatterers().size()}")
    
    bulk_solvent_mask = masks.bulk_solvent(
                xray_structure=xray_structure_p1,  # Use P1-expanded structure
                ignore_zero_occupancy_atoms=False,
                grid_step=d_min/4,  # Grid resolution (finer = more accurate)
                solvent_radius=mask_params.solvent_radius,
                shrink_truncation_radius=mask_params.shrink_truncation_radius
            )





    grid_shape = bulk_solvent_mask.data.accessor().focus()

    np_mask = np.array(bulk_solvent_mask.data).reshape(grid_shape)

    # Print mask statistics
    print(f"\nMask statistics:")
    print(f"  Grid shape: {grid_shape}")
    print(f"  Total voxels: {np_mask.size}")
    print(f"  Protein voxels (0): {np.sum(np_mask == 0)}")
    print(f"  Solvent voxels (1): {np.sum(np_mask == 1)}")
    print(f"  Solvent fraction: {np.sum(np_mask == 1) / np_mask.size * 100:.2f}%")

    cell = xray_structure.unit_cell().parameters()

    from torchref.utils import save_map


    save_map(np_mask, cell, f'/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/{prefix}bulk_mmtbx_solvent_mask.ccp4')

    M = ModelFT(verbose=2,gridsize=grid_shape).load_pdb_from_file(pdb_file)

    solvent_me = SolventModel(M)

    mask_me = solvent_me.solvent_mask.cpu().numpy()


    corrcoeff = np.corrcoef(np_mask.flatten(), mask_me.flatten())[0,1]

    save_map(mask_me, cell, f'/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/{prefix}bulk_my_solvent_mask.ccp4')

    diff = np_mask - mask_me

    save_map(diff, cell, f'/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/{prefix}bulk_solvent_mask_difference.ccp4')


    import matplotlib.pyplot as plt

    # Create 3x3 subplot figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    # Get the middle slice indices for each dimension
    nx, ny, nz = grid_shape
    slice_x = nx // 2  # Middle slice along x-axis
    slice_y = ny // 2  # Middle slice along y-axis  
    slice_z = nz // 2  # Middle slice along z-axis

    # Arrays to plot
    arrays = [diff, np_mask, mask_me]
    array_names = ['Difference (mmtbx - mine)', 'mmtbx Mask', 'My Mask']

    # Color maps
    cmaps = ['RdBu_r', 'viridis', 'viridis']

    # Plot each array in a row
    for row, (array, name, cmap) in enumerate(zip(arrays, array_names, cmaps)):
        
        # Column 0: YZ plane (slice through X at middle)
        im0 = axes[row, 0].imshow(array[slice_x, :, :].T, origin='lower', cmap=cmap, aspect='auto')
        axes[row, 0].set_title(f'{name}\nYZ plane (X={slice_x})')
        axes[row, 0].set_xlabel('Y')
        axes[row, 0].set_ylabel('Z')
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
        
        # Column 1: XZ plane (slice through Y at middle)
        im1 = axes[row, 1].imshow(array[:, slice_y, :].T, origin='lower', cmap=cmap, aspect='auto')
        axes[row, 1].set_title(f'{name}\nXZ plane (Y={slice_y})')
        axes[row, 1].set_xlabel('X')
        axes[row, 1].set_ylabel('Z')
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
        
        # Column 2: XY plane (slice through Z at middle)
        im2 = axes[row, 2].imshow(array[:, :, slice_z].T, origin='lower', cmap=cmap, aspect='auto')
        axes[row, 2].set_title(f'{name}\nXY plane (Z={slice_z})')
        axes[row, 2].set_xlabel('X')
        axes[row, 2].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    # Overall title
    fig.suptitle('Bulk Solvent Mask Comparison: Middle Slices Through Three Orthogonal Planes', 
                fontsize=16, fontweight='bold', y=0.995)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    output_path = f'/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/{prefix}mask_comparison_slices.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    return corrcoeff



dilation = []
corr = []

for dilation_radius in [1]:
    print(f"\n\n=== Testing P21 Symmetry with dilation radius {dilation_radius} ===")
    dilation_str = str(dilation_radius).replace('.','p')
    corr_P21 = test_for_pdb_file(pdb_file_P21,prefix=f'P21_dilation_{dilation_str}_',dilation_radius=dilation_radius)
    print(f"Correlation coefficient between masks (P21, dilation {dilation_radius}): {corr_P21:.6f}")
    corr.append(corr_P21)
    dilation.append(dilation_radius)

print(corr)
print(dilation)

