import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from cifar10.mothers.graphs import test_vs_pred, num_train_vs_test_graph
import numpy as np
import json

model = nn.Sequential(
    nn.Linear(396, 256),
    nn.Sigmoid(),
    nn.Linear(256, 128),
    nn.Sigmoid(),
    nn.Linear(128, 1)
)


def train_model():
    with open('../data/sparse_perlin.txt', 'r') as f:
        train_data = [json.loads(line) for line in f]

    X = torch.tensor(np.array([np.array(data['density_map']).flatten() for data in train_data]), dtype=torch.float32)
    y = torch.tensor([data['test_losses'][-1] for data in train_data], dtype=torch.float32)

    X_train, X_test, y_train, y_test, jsons_train, jsons_test = train_test_split(X, y, train_data, test_size=0.2, random_state=42)
    y_train, y_test = y_train.unsqueeze(1), y_test.unsqueeze(1)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    print(X_train.shape)

    # Train Model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps")
    model.to(device)
    epochs = 100

    for epoch in range(epochs):
        losses = []
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            output = model.forward(data)
            loss = criterion(output, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Epoch: {sum(losses) / len(losses)}')


    torch.save(model.state_dict(), './feed_forward_new.pth')
    y_pred = model.forward(X_test.to(device))
    loss = criterion(y_pred, y_test.to(device))

    print(loss.item())

    # Graph
    y_pred = y_pred.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()

    num_train_vs_test_graph(
        y_test, y_pred, jsons_test, loss.item(),
        show=True
    )

    test_vs_pred(
        y_test, y_pred, loss, show=True,
        ylim=(None, None),
        xlim=(None, None)
    )


if __name__ == '__main__':
    train_model()