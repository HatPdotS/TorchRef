from torchref.kinetics import KineticModel
import torch
import matplotlib.pyplot as plt

test_cases = [
    ("A->B,B->C", 3, 2),
    ("A->B,B->A", 2, 2),
    ("A->B,A->C", 3, 2),
    ("A->B,B->C,B->A", 3, 3),
    ("A->B,B->C,C->D,C->A", 4, 4),
]


timepoints = torch.linspace(0,100,10000) 

flow_chart = 'A->B,B->C,A->F,C->D,D->E,E->A'

kinetic_model = KineticModel(
    flow_chart=flow_chart,
    timepoints=timepoints,instrument_width=20)

kinetic_model.set_baseline(state='A', occupancy=0)

kinetic_model.print_parameters()



kinetic_model.plot_occupancies('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/kinetics/rubisco_like.png')

plt.close()

times = torch.logspace(0, 4,1000)

kinetic_model.plot_occupancies('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/kinetics/rubisco_like_long_log.png',times=times, log=True)