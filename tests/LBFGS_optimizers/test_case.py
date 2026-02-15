import torch



def hard_to_optimize(x):
    return torch.sin(5 * (x - 4) ) + 0.01 * (x - 4) ** 2


def plot(x_optimized):
    from matplotlib import pyplot as plt

    x = torch.linspace(-10, 10, 1000)
    y = hard_to_optimize(x)
    plt.plot(x.detach().numpy(), y.detach().numpy())
    plt.vlines(x_optimized.detach().numpy(), ymin=-1, ymax=1, color='red', label='Optimized Value')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Hard to Optimize Function')
    plt.savefig('/das/work/p17/p17490/Peter/Library/torchref/tests/LBFGS_optimizers/hard_to_optimize_function.png')



starting_point = torch.tensor(0.0, requires_grad=True)
optimizer = torch.optim.LBFGS([starting_point], max_iter=100, line_search_fn='strong_wolfe')

def closure():
    optimizer.zero_grad()
    loss = hard_to_optimize(starting_point)
    loss.backward()
    return loss

for i in range(100):
    optimizer.step(closure)

print(f"Optimized x: {starting_point.item()}")
print(f"Optimized f(x): {hard_to_optimize(starting_point).item()}")

plot(starting_point)