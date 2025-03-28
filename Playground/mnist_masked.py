import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from Sequential2D import Sequential2D, MaskedLinear, SparseAdam
from util import num_trainable_parameters
import numpy as np
import time


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
                MaskedLinear.sparse_random(sizes[j], sizes[i], percent=1),
            )

#            2500  500   200   100   10
# blocks = [[I,    None, None, None, None],
#           [f10,  f11,  f12,  f13,  f14 ],
#           [f20,  f21,  f22,  f23,  f24 ],
#           [f30,  f31,  f32,  f33,  f34 ],
#           [f40,  f41,  f42,  f43,  f44 ]]


model = Sequential2D(blocks)
print(f'Trainable: {num_trainable_parameters(model)}')

# train
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
optimizer = SparseAdam(model.parameters(), lr=0.0001)

for epoch in range(100):

    losses = []
    forward_times = []
    backward_times = []

    for images, labels in train_loader:
        batch_size = images.shape[0]

        start = time.time()  # TIMER START

        output = model.forward([
            images.view(batch_size, -1),
            torch.zeros(batch_size, 500),
            torch.zeros(batch_size, 200),
            torch.zeros(batch_size, 100),
            torch.zeros(batch_size, 10)
        ])
        output = model.forward([F.relu(x) for x in output])
        output = model.forward([F.relu(x) for x in output])
        output = model.forward([F.relu(x) for x in output])

        forward_times.append(time.time() - start)  # TIMER END

        start = time.time()  # TIMER START

        loss = criterion(output[4], labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        backward_times.append(time.time() - start)  # TIMER END

        print(f'Loss: {sum(losses) / len(losses)}')
        print(f'Forward:  {sum(forward_times) / len(forward_times)}')
        print(f'Backward: {sum(backward_times) / len(backward_times)}')

    if (epoch - 1) % 1 == 0:
        print(f'{sum(losses) / len(losses)}')