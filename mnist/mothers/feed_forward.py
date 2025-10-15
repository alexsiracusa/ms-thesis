import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from mnist.mothers import load_dataset
from mnist.util import get_num_trainable
from sklearn.model_selection import train_test_split
from mnist.mothers.graphs import num_train_vs_test_graph, test_vs_pred
from mnist.datasets import datasets

include = list(datasets.keys())[:-3]
super_include = ['blood_mnist', 'chinese_mnist']
include = set(include) - set(super_include)

params = {
    'noise_types': ['sparse_perlin'],
    'feature_set': ['density_map'],
    'dataset_feature_set': ['ce_loss'],
    'target': 'test_loss',
    'min_cut_off': 0,
    'max_cut_off': 1,
    'max_target': 5,
}

features, targets, jsons = load_dataset(**params, include=include)
super_features, super_targets, super_jsons = load_dataset(**params, include=super_include)
super_targets = super_targets.unsqueeze(1)

X_train, X_test, y_train, y_test, jsons_train, jsons_test = train_test_split(features, targets, jsons, test_size=0.2, random_state=42)
y_train, y_test = y_train.unsqueeze(1), y_test.unsqueeze(1)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

print(X_train.shape)

model = nn.Sequential(
    nn.Linear(397, 256),
    nn.Sigmoid(),
    nn.Linear(256, 128),
    nn.Sigmoid(),
    nn.Linear(128, 1)
)

# Train Model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps")
model.to(device)
epochs = 25

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


# y_pred = model.forward(X_test.to(device))
# loss = criterion(y_pred, y_test.to(device))
#
# print(loss.item())
#
# # Graph
# y_pred = y_pred.detach().cpu().numpy()
# y_test = y_test.detach().cpu().numpy()
#
# num_train_vs_test_graph(
#     y_test, y_pred, jsons_test, loss.item(),
#     show=True
# )
#
# test_vs_pred(
#     y_test, y_pred, loss, show=True,
#     ylim=(None, None),
#     xlim=(None, None)
# )


# Super test
y_pred = model.forward(super_features.to(device))
loss = criterion(y_pred, super_targets.to(device))

print(loss.item())

# Graph
y_pred = y_pred.detach().cpu().numpy()
y_test = super_targets.detach().cpu().numpy()

num_train_vs_test_graph(
    y_test, y_pred, super_jsons, loss.item(),
    show=True
)

test_vs_pred(
    y_test, y_pred, loss, show=True,
    ylim=(None, None),
    xlim=(None, None)
)