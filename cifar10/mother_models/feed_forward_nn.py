import numpy as np
import random
import json

import torch
import torch.nn as nn

random.seed(0)

with open('../train_epoch=3/train_data.txt', 'r') as f:
    train_data = [json.loads(line) for line in f]
    random.shuffle(train_data)
    train_cut = int(0.8 * len(train_data))

X = [np.array(data['densities']).flatten() for data in train_data]
y = [data['test_loss'] for data in train_data]

X_train, y_train = X[:train_cut], y[:train_cut]
X_test, y_test = X[train_cut:], y[train_cut:]

model = nn.Sequential(
    nn.Linear(6525, 2000),
    nn.ReLU(),
    nn.Linear(2000, 500),
    nn.ReLU(),
    nn.Linear(500, 1),
    nn.ReLU()
)


