import json
import numpy as np
import random

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def load_data(path):
    with open(path, 'r') as f:
        train_data = [json.loads(line) for line in f]
        random.shuffle(train_data)
        train_cut = int(0.8 * len(train_data))

    singular_values = np.array([
        np.linalg.svd(np.array(data['densities']), compute_uv=False)
        for data in train_data
    ])

    X = np.array([np.array(data['densities']).flatten() for data in train_data])
    # X = np.concatenate((X, singular_values), axis=1)
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

    return train_data, train_cut, train_loader, X_train, y_train, X_test, y_test


if __name__ == '__main__':
    load_data('../train_epoch=3/perlin_data.txt')
