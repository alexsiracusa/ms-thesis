from Sequential2D import IterativeSequential2D, FlatIterativeSequential2D
import torch.nn.functional as F
import torch


I = torch.nn.Identity()
#          2     2     2     2
blocks = [[I, None, None, None],  # 2
          [None, I, None, None],  # 2
          [I, I, I, I],  # 2
          [I, I, I, I]]  # 2
sizes = [2, 2, 2, 2]

model = FlatIterativeSequential2D(blocks, sizes, num_iterations=2)

X = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
y = model.forward(X)
assert (y.allclose(torch.tensor([[1, 1, 1, 1, 10, 10, 10, 10]])))

X = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
y = model.forward(X)
assert (y.allclose(torch.tensor([[1, 1, 1, 1, 6, 6, 6, 6]])))

X = torch.tensor([[1, 1, 1, 1]])
y = model.forward(X)
assert (y.allclose(torch.tensor([[1, 1, 1, 1, 6, 6, 6, 6]])))


model = FlatIterativeSequential2D(blocks, sizes, num_iterations=1, activations=[I, I, I, F.relu])

X = torch.tensor([[-1, -1, -1, -1, -1, -1, -1, -1]])
y = model.forward(X)
assert (y.allclose(torch.tensor([[-1, -1, -1, -1, -4, -4, 0, 0]])))






