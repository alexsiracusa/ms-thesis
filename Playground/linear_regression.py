from Sequential2D import Sequential2D
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from util import generate_linear_data


num_features = 8

# define model
I = torch.nn.Identity()
f1 = torch.nn.Linear(in_features=8, out_features=4)
f2 = torch.nn.Linear(in_features=4, out_features=2)
f3 = torch.nn.Linear(in_features=2,    out_features=1)

#          8     4     2     1
blocks = [[I,    None, None, None], # 8
          [f1,   None, None, None], # 4
          [None, f2,   None, None], # 2
          [None, None, f3,   None]] # 1

model = Sequential2D(blocks)


# generate dataset
np.random.seed(0)
X, y = generate_linear_data(100, num_features, noise_std=0.4)
X = torch.from_numpy(X).to(torch.float32)
y = torch.from_numpy(y).to(torch.float32)


# train
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500 + 1):
    output = model.forward([X, torch.zeros(100, 4), torch.zeros(100, 2), torch.zeros(100, 1)])
    output = model.forward(output)
    output = model.forward(output)

    loss = criterion(output[3], y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f'Loss: {loss}')



