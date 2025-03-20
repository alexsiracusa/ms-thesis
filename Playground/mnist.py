import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from Sequential2D import Sequential2D


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


# define model
I = nn.Identity()
f1 = nn.Sequential(
    torch.nn.Linear(in_features=2500, out_features=500),
    torch.nn.ReLU()
)
f2 = nn.Sequential(
    torch.nn.Linear(in_features=500, out_features=200),
    torch.nn.ReLU()
)
f3 = nn.Sequential(
    torch.nn.Linear(in_features=200, out_features=100),
    torch.nn.ReLU()
)
f4 = nn.Sequential(
    torch.nn.Linear(in_features=100, out_features=10),
    torch.nn.ReLU()
)

#          2500  500   200   100   10
blocks = [[I,    None, None, None, None],
          [f1,   None, None, None, None],
          [None, f2,   None, None, None],
          [None, None, f3,   None, None],
          [None, None, None, f4,   None]]

model = Sequential2D(blocks)

# train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(100):

    losses = []

    for images, labels in train_loader:
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

        loss = criterion(output[4], labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if (epoch - 1) % 1 == 0:
        print(f'Loss: {sum(losses) / len(losses)}')