import torch
import torch.nn as nn
import torch.optim as optim

from load_cifar import load_cifar
from flatten_images import flatten_images
from Sequential2D.util import build_sequential2d


data_folder = "../data"
train_loader, test_loader = load_cifar(data_folder, batch_size=128)

# 12104210
# 16575000

input_sizes = [75] * 100
hidden_sizes = [100] * 22
output_sizes = [10]

# input_sizes = [7500]
# hidden_sizes = [2200]
# output_sizes = [10]

sizes = input_sizes + hidden_sizes + output_sizes
print(sum(sizes))

model = build_sequential2d(
    sizes,
    type='flat',
    num_input_blocks=len(input_sizes),
    num_output_blocks=len(output_sizes),
    num_iterations=4,
    densities=0.73,
    weight_init='weighted',
)

criterion = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

model.to(device)

for epoch in range(num_epochs):
    losses = []
    for images, labels in train_loader:
        images = flatten_images(images, kernel_size=5, stride=5)
        images = images.to(device)
        labels = labels.to(device)

        output = model.forward(images)[:, -10:]
        loss = criterion(output, labels)

        loss.backward()
        optim.step()
        optim.zero_grad()

        losses.append(loss.item())
        print(f'Loss: {loss.item():.5f}')

    print(f'Epoch {epoch}/{num_epochs} Loss: {sum(losses) / len(losses):.5f}')


