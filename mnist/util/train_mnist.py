import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from mnist.util import load_mnist, create_model
from mnist.util.random_densities import sparse_perlin


def average_loss(model, loader, device, output_size):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    losses = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model.forward(images)[:, -output_size:]
            loss = criterion(output, labels)
            losses.append(loss.item())

    return sum(losses) / len(losses)


def train_mnist(
        model,
        train_loader,
        test_loader,
        epochs=3,
        output_size=10,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    print(f'Device: {device}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)

    # train loop
    for epoch in range(epochs):
        losses = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model.forward(images)[:, -output_size:]
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        loss = sum(losses) / len(losses)
        print(f'Epoch {epoch + 1}/{epochs} Loss: {loss:.5f}')

        try:
            wandb.log({"epoch_loss": loss})
        except:
            pass

    # final training loss
    train_loss = average_loss(model, train_loader, device, output_size=output_size)
    test_loss = average_loss(model, test_loader, device, output_size=output_size)
    print(f'train: {train_loss:.5f}, test: {test_loss:.5f}')

    return train_loss, test_loss


if __name__ == '__main__':
    from mnist.util.sizes import num_blocks, num_input

    train_loader, test_loader = load_mnist(
        '../../data',
        dataset='MNIST',
        flatten=True,
        batch_size=128
    )

    densities = sparse_perlin((num_blocks - num_input, num_blocks), clip=0.33)
    model = create_model(densities)

    train_mnist(model, train_loader, test_loader, device=torch.device("mps"), epochs=2, output_size=10)