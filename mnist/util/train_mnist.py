import torch
import torch.nn as nn
import torch.optim as optim

from mnist.util import load_mnist


def average_loss(model, loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    losses = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model.forward(images)[:, -10:]
            loss = criterion(output, labels)
            losses.append(loss.item())

    return sum(losses) / len(losses)


def train_mnist(
        model,
        train_loader,
        test_loader,
        epochs=3
):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    model.to(device)

    # train loop
    for epoch in range(epochs):
        losses = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model.forward(images)[:, -10:]
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        print(f'Epoch {epoch + 1}/{epochs} Loss: {sum(losses) / len(losses):.5f}')

    # final training loss
    train_loss = average_loss(model, train_loader, device)
    test_loss = average_loss(model, test_loader, device)
    print(f'train: {train_loss:.5f}, test: {test_loss:.5f}')

    return train_loss, test_loss


if __name__ == '__main__':
    from Sequential2D.util import build_sequential2d
    from mnist.util.sizes import *

    train_loader, test_loader = load_mnist(
        '../data',
        dataset='MNIST',
        flatten=True,
        batch_size=128
    )

    model = build_sequential2d(
        sizes,
        type='linear',
        num_input_blocks=len(input_sizes),
        num_output_blocks=len(output_sizes),
        num_iterations=4,
        densities=0.5,
        weight_init='weighted',
    )

    train_mnist(model, train_loader, test_loader)