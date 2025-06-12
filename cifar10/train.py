import torch
import torch.nn as nn
import torch.optim as optim

from load_cifar import load_cifar
from flatten_images import flatten_images
from Sequential2D.util import build_sequential2d


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


def train_cifar(input_sizes, hidden_sizes, output_sizes, num_iterations, densities):

    sizes = input_sizes + hidden_sizes + output_sizes
    print(sum(sizes))

    model = build_sequential2d(
        sizes,
        type='linear',
        num_input_blocks=len(input_sizes),
        num_output_blocks=len(output_sizes),
        num_iterations=num_iterations,
        densities=densities.tolist(),
        weight_init='weighted',
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    model.to(device)

    # train loop
    for epoch in range(num_epochs):
        losses = []
        for images, labels in train_loader:
            images = flatten_images(images, kernel_size=5, stride=5)
            images = images.to(device)
            labels = labels.to(device)

            output = model.forward(images)[:, -10:]
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            # print(f'Loss: {loss.item():.5f}')

        print(f'Epoch {epoch+1}/{num_epochs} Loss: {sum(losses) / len(losses):.5f}')

    # final training loss
    train_loss = average_loss(model, train_loader, device)
    test_loss = average_loss(model, test_loader, device)

    return train_loss, test_loss


if __name__ == '__main__':
    data_folder = "../data"
    train_loader, test_loader = load_cifar(data_folder, batch_size=128)

    input_sizes = [75] * 100
    hidden_sizes = [50] * 44
    output_sizes = [10]

    num_blocks = len(input_sizes + hidden_sizes + output_sizes)

    densities = torch.empty((num_blocks, num_blocks)).uniform_(0, 1)
    num_iterations = 4

    train_loss, test_loss = train_cifar(input_sizes, hidden_sizes, output_sizes, num_iterations, densities)

    print(f'Final train loss: {train_loss:.5f}')
    print(f'Final test loss:  {test_loss:.5f}')


