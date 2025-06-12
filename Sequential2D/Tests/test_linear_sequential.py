import torch
from Sequential2D import LinearSequential2D

sizes = [2, 1, 2, 1]
num_input_blocks = 2
densities = [
    None,
    None,
    [1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
]

linear = LinearSequential2D(sizes, num_input_blocks=num_input_blocks, bias=False, densities=densities)
X = torch.tensor([[1, 1, 1]], dtype=torch.float32)
y = linear.forward(X)

expected_mask = torch.tensor([
    [1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1.],
    [0., 0., 0., 0., 0., 0.],
])

assert torch.equal(expected_mask, linear.linear.mask)
assert torch.equal(X[0, :sum(sizes[:num_input_blocks])], y[0, :sum(sizes[:num_input_blocks])])
assert y[0, -1] == 0

print(linear.linear.mask)
print(y)
