from mnist.datasets import datasets
from mnist.datasets import datasets, load_parquet
from torch.utils.data import DataLoader, TensorDataset
from mnist.util import train_mnist, create_model, sparse_perlin, flatten_images, sparse_random

import torch
import torch.nn as nn
import torch.optim as optim


def average_loss(model, loader, device):
    model.to(device)
    criterion = nn.MSELoss()

    losses = []

    with torch.no_grad():
        for images in loader:
            images = images[0].to(device)

            output = model.forward(images)
            loss = criterion(output, images)
            losses.append(loss.item())

    return sum(losses) / len(losses)

# List of datasets
datasets = list(datasets.keys())

for dataset in datasets:
    print(f"---- Dataset: {dataset} ----")

    train_images, _ = load_parquet(f"../datasets/parquets/{dataset}/train.parquet")
    test_images, _ = load_parquet(f"../datasets/parquets/{dataset}/train.parquet")

    train_images = (train_images - train_images.min()) / (train_images.max() - train_images.min())
    test_images = (test_images - test_images.min()) / (test_images.max() - test_images.min())

    train_images = flatten_images(train_images, kernel_size=(10, 10), stride=10, padding=0)
    test_images = flatten_images(test_images, kernel_size=(10, 10), stride=10, padding=0)

    train_dataset = TensorDataset(train_images)
    test_dataset = TensorDataset(test_images)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # print(dataset, train_images.shape)

    model = nn.Sequential(
        nn.Linear(2500, 500),
        nn.ReLU(),
        nn.Linear(500, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 128),
        nn.ReLU(),
        nn.Linear(128, 500),
        nn.ReLU(),
        nn.Linear(500, 2500),
        nn.Sigmoid()
    )

    epochs = 5
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)

    # train loop
    for epoch in range(epochs):
        losses = []
        for images in train_loader:
            images = images[0].to(device)

            output = model.forward(images)
            loss = criterion(output, images)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        loss = sum(losses) / len(losses)
        print(f'Epoch {epoch + 1}/{epochs} Loss: {loss:.5f}')

    # final training loss
    train_loss = average_loss(model, train_loader, device)
    test_loss = average_loss(model, test_loader, device)

    print(f'train: {train_loss:.5f}, test: {test_loss:.5f}')
    print("\n")


