import torch.nn as nn
import torch.nn.functional as F

from Sequential2D import IterativeSequential2D


# define model
I = nn.Identity()
f1 = nn.Linear(in_features=7500, out_features=1500)
f2 = nn.Linear(in_features=1500, out_features=500)
f3 = nn.Linear(in_features=500, out_features=200)
f4 = nn.Linear(in_features=200, out_features=10)

#          7500  1500  500   200   10
blocks = [[I,    None, None, None, None],
          [f1,   None, None, None, None],
          [None, f2,   None, None, None],
          [None, None, f3,   None, None],
          [None, None, None, f4,   None]]

model = IterativeSequential2D(blocks, 4, F.relu)

if __name__ == '__main__':
    print(sum(p.numel() for p in model.parameters()))
