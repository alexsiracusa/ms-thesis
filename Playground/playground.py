from Sequential2D import Sequential2D
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from util import generate_linear_data


I = torch.nn.Identity()
f1 = torch.nn.Sequential(
    torch.nn.Linear(in_features=8, out_features=4),
    # torch.nn.ReLU()
)

f2 = torch.nn.Sequential(
    torch.nn.Linear(in_features=4, out_features=2),
    # torch.nn.ReLU()
)

f3 = torch.nn.Linear(in_features=2,    out_features=1)

#          8     4     2     1
blocks = [[I,    None, None, None], # 8
          [f1,   None, None, None], # 4
          [None, f2,   None, None], # 2
          [None, None, f3,   None]] # 1

model = Sequential2D(blocks)


# X = [torch.zeros(8), torch.zeros(4), torch.zeros(2), torch.zeros(1)]

np.random.seed(0)
torch.manual_seed(0)

X, y = generate_linear_data(1000, 8, noise_std=0)
X = torch.from_numpy(X).to(torch.float32)
y = torch.from_numpy(y).to(torch.float32)


# train
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00002)

for epoch in range(10000):
    output = model.forward([X, torch.zeros(1000, 4), torch.zeros(1000, 2), torch.zeros(1000, 1)])
    output = model.forward(output)
    output = model.forward(output)

    loss = criterion(output[3], y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Loss: {loss}')



# for tensor in X:
#     print(tensor.detach().numpy())

