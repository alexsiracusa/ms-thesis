import numpy as np
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from util import train
from cifar10.util import get_num_trainable

import matplotlib.pyplot as plt


random.seed(0)

# LOAD DATA
with open('../train_epoch=3/train_data.txt', 'r') as f:
    train_data = [json.loads(line) for line in f]
    random.shuffle(train_data)
    train_cut = int(0.8 * len(train_data))

X = np.array([np.array(data['densities']).flatten() for data in train_data])
y = np.array([[data['test_loss']] for data in train_data])

X_train, y_train = X[:train_cut], y[:train_cut]
X_test, y_test = X[train_cut:], y[train_cut:]

X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

training_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    training_dataset,
    batch_size=64,
    shuffle=True
)


# TRAIN MODEL
model = nn.Sequential(
    nn.Linear(6525, 2000),
    nn.ReLU(),
    nn.Linear(2000, 500),
    nn.ReLU(),
    nn.Linear(500, 1),
    nn.ReLU()
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
train(model, train_loader, criterion, optimizer, epochs=3)


# EVALUATE MODEL
y_pred = model.forward(X_test)
loss = criterion(y_pred, y_test)

print(loss.item())

num_trainable = [get_num_trainable(data['densities']) for data in train_data][train_cut:]
plt.scatter(num_trainable, y_test.detach().numpy())
plt.scatter(num_trainable, y_pred.detach().numpy())
plt.savefig('graph.png')



