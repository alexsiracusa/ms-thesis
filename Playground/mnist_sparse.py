import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from Sequential2D import Sequential2D, SparseLinear, SparseAdam
from util import num_trainable_parameters
import numpy as np


def load_mnist(data_folder):
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
    ])

    train_dataset = MNIST(root=data_folder, train=True, transform=transform, download=True)
    test_dataset = MNIST(root=data_folder, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

data_folder = "../data"
train_loader, test_loader = load_mnist(data_folder)


# normal trainable: 1371810
# full trainable:   2685150

sizes = [2500, 500, 200, 100, 10]
blocks = np.empty((len(sizes), len(sizes)), dtype=object)

for i in range(len(sizes)):
    for j in range(len(sizes)):
        if i == 0 and j == 0:
            blocks[i, j] = torch.nn.Identity()
        elif i == 0:
            blocks[i, j] = None
        else:
            blocks[i, j] = nn.Sequential(
                SparseLinear.sparse_random(sizes[j], sizes[i], percent=0.5108),
                torch.nn.ReLU()
            )

#            2500  500   200   100   10
# blocks = [[I,    None, None, None, None],
#           [f10,  f11,  f12,  f13,  f14 ],
#           [f20,  f21,  f22,  f23,  f24 ],
#           [f30,  f31,  f32,  f33,  f34 ],
#           [f40,  f41,  f42,  f43,  f44 ]]

device = torch.device('cpu')
model = Sequential2D(blocks)
model.to(device)
print(f'Trainable: {num_trainable_parameters(model)}')


# train
criterion = nn.CrossEntropyLoss()
optimizer = SparseAdam(model.parameters(), lr=0.0001)
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(100):

    losses = []

    print(len(train_loader))
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.shape[0]

        output = model.forward([
            images.view(batch_size, -1),
            torch.zeros(batch_size, 500),
            torch.zeros(batch_size, 200),
            torch.zeros(batch_size, 100),
            torch.zeros(batch_size, 10)
        ])
        output = model.forward(output)
        output = model.forward(output)
        output = model.forward(output)
        # output = model.forward([F.relu(x) for x in output])
        # output = model.forward([F.relu(x) for x in output])
        # output = model.forward([F.relu(x) for x in output])

        loss = criterion(output[4], labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'Loss: {sum(losses) / len(losses)}')

    if (epoch - 1) % 1 == 0:
        print(f'Loss: {sum(losses) / len(losses)}')