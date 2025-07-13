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

X = np.array([np.array([data['densities']]) for data in train_data])
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
                                                                          # 1 x 45 x 145
    nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1),    # 5 x 43 x 143
    nn.MaxPool2d(kernel_size=2, stride=2),                                # 5 x 21 x 71
    nn.ReLU(),

    nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, stride=1),   # 10 x 19 x 69
    nn.ReLU(),
    nn.Conv2d(in_channels=10, out_channels=15, kernel_size=3, stride=1),  # 15 x 17 x 67
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),                                # 15 x 8 x 33

    nn.Conv2d(in_channels=15, out_channels=10, kernel_size=(2, 3), stride=(1, 3)),  # 10 x 7 x 11
    nn.ReLU(),

    nn.Flatten(),
    nn.Linear(10 * 7 * 11, 128),
    nn.ReLU(),
    nn.Linear(128, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(model, train_loader, criterion, optimizer, epochs=50, device=device)


# EVALUATE MODEL
y_pred = model.forward(X_test.to(device))
loss = criterion(y_pred, y_test.to(device))

print(loss.item())

num_trainable = [get_num_trainable(data['densities']) for data in train_data][train_cut:]
plt.scatter(num_trainable, y_test.detach().cpu().numpy())
plt.scatter(num_trainable, y_pred.detach().cpu().numpy())
plt.figtext(0.5, 0.5, f'Loss: {loss.item():.7f}', fontsize=12, color='red')
plt.savefig('cnn.png')




