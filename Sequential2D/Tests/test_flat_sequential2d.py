from Sequential2D import FlatSequential2D
import torch.nn.functional as F
import torch

I = torch.nn.Identity()
#          2     2     2     2
blocks = [[I, None, None, None],  # 2
          [None, I, None, None],  # 2
          [I, I, I, I],  # 2
          [I, I, I, I]]  # 2

model = FlatSequential2D(
    in_features=[2, 2, 2, 2],
    out_features=[2, 2, 2, 2],
    blocks=blocks
)

X = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
y = model.forward(X)
assert (y.allclose(torch.tensor([[1, 1, 1, 1, 4, 4, 4, 4]])))

y = model.forward(y)
assert (y.allclose(torch.tensor([[1, 1, 1, 1, 10, 10, 10, 10]])))

X = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]])
y = model.forward(X)
assert (y.allclose(torch.tensor([[1, 1, 1, 1, 4, 4, 4, 4], [0, 0, 0, 0, 0, 0, 0, 0]])))

