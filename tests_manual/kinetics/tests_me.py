from torchref.kinetics import KineticModel
import torch


test_cases = [
    ("A->B,B->C", 3, 2),
    ("A->B,B->A", 2, 2),
    ("A->B,A->C", 3, 2),
    ("A->B,B->C,B->A", 3, 3),
    ("A->B,B->C,C->D,C->A", 4, 4),
]


timepoints = torch.linspace(0,100,10000) 

kinetic_model = KineticModel(
    flow_chart="A->B,B->C,C->D,C->A,B->A",
    timepoints=timepoints,instrument_width=20)

kinetic_model.print_parameters()

kinetic_model.plot_occupancies('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/kinetics/kinetics_test_occupancies.png')