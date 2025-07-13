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

X = [[get_num_trainable(data['densities']) / 1e7] for data in train_data]
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

size = 64
model = nn.Sequential(
    nn.Linear(1, size),
    nn.ReLU(),
    nn.Linear(size, size),
    nn.ReLU(),
    nn.Linear(size, 1),
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(model, train_loader, criterion, optimizer, epochs=100, device=device)


# EVALUATE MODEL
y_pred = model.forward(X_test.to(device))
loss = criterion(y_pred, y_test.to(device))

print(loss.item())

num_trainable = [get_num_trainable(data['densities']) for data in train_data][train_cut:]
plt.scatter(num_trainable, y_test.detach().cpu().numpy())
plt.scatter(num_trainable, y_pred.detach().cpu().numpy())
plt.figtext(0.5, 0.5, f'Loss: {loss.item():.7f}', fontsize=12, color='red')
plt.savefig('mini.png')




