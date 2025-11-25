from torchref.math_functions.math_torch import find_relevant_voxels
import torch


x = torch.arange(0,10,step=0.1)
y = torch.arange(0,10,step=0.1)
z = torch.arange(0,10,step=0.1)

grid = torch.meshgrid(x, y, z, indexing='ij')
real_space_grid = torch.stack(grid, dim=-1)  # Shape: (10, 10, 10, 3)


points = torch.tensor([[-4.5, 1.5, 1.5],
                       [8.5, 8.5, 8.5],
                       [5.0, 5.0, 5.0]])

points, indices = find_relevant_voxels(real_space_grid, points, radius=3)

print("Points:\n", points)
print("Indices:\n", indices)