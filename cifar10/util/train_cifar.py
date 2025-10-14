import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from cifar10.util import load_cifar
from cifar10.util import flatten_images


def average_loss(model, loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    losses = []

    with torch.no_grad():
        for images, labels in loader:
            images = flatten_images(images, kernel_size=5, stride=5)
            images = images.to(device)
            labels = labels.to(device)

            output = model.forward(images)[:, -10:]
            loss = criterion(output, labels)
            losses.append(loss.item())

    return sum(losses) / len(losses)


def train_cifar(
        model, train_loader, test_loader,
        epochs=3,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    print(f'Device: {device}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    epoch_losses = []
    train_losses = []
    test_losses = []

    # train loop
    for epoch in range(epochs):
        losses = []
        for images, labels in train_loader:
            images = flatten_images(images, kernel_size=10, stride=10, padding=0)
            images = images.to(device)
            labels = labels.to(device)

            output = model.forward(images)[:, -10:]
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        # Evaluate every epoch
        epoch_loss = sum(losses) / len(losses)
        train_loss = average_loss(model, train_loader, device)
        test_loss = average_loss(model, test_loader, device)

        epoch_losses.append(epoch_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        try:
            wandb.log({
                'epoch_loss': epoch_loss,
                'train_loss': train_loss,
                'test_loss': test_loss,
            })
        except:
            pass

        print(f'Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.5f}')

    # final training loss
    train_loss = train_losses[-1]
    test_loss = test_losses[-1]
    print(f'train: {train_loss:.5f}, test: {test_loss:.5f}')

    return train_losses, test_losses, epoch_losses


if __name__ == '__main__':
    from cifar10.util import sparse_perlin, create_model
    from cifar10.util.sizes import num_blocks, num_input
    import matplotlib.pyplot as plt

    data_folder = "../../data"
    train_loader, test_loader = load_cifar(data_folder, batch_size=128)

    from mnist.util.sizes import num_blocks, num_input

    densities = sparse_perlin((num_blocks - num_input, num_blocks), clip=0.33)
    model = create_model(densities)

    plt.imshow(densities, cmap='grey')
    plt.show()

    train_cifar(model, train_loader, test_loader, device=torch.device("mps"), epochs=3)


