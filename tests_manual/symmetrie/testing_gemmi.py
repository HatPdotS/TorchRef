import gemmi
from torchref.symmetrie import SYMMETRY_OPERATIONS, SPACEGROUP_NAME_MAPPING
import torch




def test_for_spacegroup(sg_name):
    """Print symmetry operations for a given space group using gemmi"""
    try:
        gemmi_sg = gemmi.SpaceGroup(sg_name)
    except Exception as e:
        print(f"Failed to find space group {sg_name} in gemmi: {e}")
        return
    
    gemmi_ops = [(torch.tensor(i.rot)/24, torch.tensor(i.tran)/24) for i in gemmi_sg.operations()]

    canonical = SPACEGROUP_NAME_MAPPING.get(sg_name, sg_name)
    our_ops = SYMMETRY_OPERATIONS.get(canonical, None)
    our_ops = [(our_ops[0][i], our_ops[1][i]) for i in range(our_ops[0].shape[0])] if our_ops is not None else None
    if our_ops is None:
        print('did not find our operations for', canonical)
        return
    assert len(our_ops) == len(gemmi_ops), f"Operation count mismatch for SG {sg_name}"
    for operation in gemmi_ops:
        # Find matching operation in our implementation
        match_found = False
        for our_op in our_ops:
            if torch.allclose(operation[0], our_op[0]) and torch.allclose(operation[1], our_op[1]):
                match_found = True
                break
        if not match_found:
            print("No match found for operation:")
            print("Rotation:\n", operation[0].numpy())
            print("Translation:\n", operation[1].numpy())
            raise ValueError("No matching operation found in our implementation for SG " + sg_name)
    return 0





spacegroups = ['P1', 'P2', 'P21', 'C2', 'Pm', 'Pc', 'Cm', 'Cc', 'P2/m', 'P21/m', 'C2/m',
               'Pmcm', 'P4', 'P41', 'P42', 'P43', 'I4', 'I41', 'P-4', 'I-4',
               'P3', 'P31', 'P32', 'R3', 'P6', 'P61', 'P62', 'P63', 'P64', 'P65']
for sg in spacegroups:
    print(f"\n{'='*60}\nTesting space group: {sg}\n{'='*60}")
    test_for_spacegroup(sg)