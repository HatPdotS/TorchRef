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

timepoints = torch.logspace(-9,0,10000) 

Br_rate_constants = {
    'A->K': 1e10,
    'K->L': 5e5,
    'L->M': 2.5e4,
    'M->N': 200,
    'N->O':200,
    'O->A':200}

flow_chart = 'A->K,K->L,L->M,M->N,N->O,O->A'

kinetic_model = KineticModel(
    flow_chart=flow_chart,
    timepoints=timepoints,
    instrument_width=1e-8,
    rate_constants=Br_rate_constants,
    light_activated=True  # Prevent re-photoactivation: O->A* instead of O->A
)
kinetic_model.set_baseline(state='A', occupancy=0.2)

kinetic_model.print_parameters()


kinetic_model.plot_occupancies('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/kinetics/BR_like.png')

plt.close()

kinetic_model.plot_occupancies('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/kinetics/BR_like_long_log.png', log=True)


print(kinetic_model.parameters())