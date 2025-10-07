import random

import torch
import torch.nn as nn
import torch.optim as optim

from util import train
from cifar10.util import get_num_trainable

import matplotlib.pyplot as plt

from cifar10.mother_models.load_data import load_data


model = nn.Sequential(
    nn.Linear(6525, 2000),
    nn.Sigmoid(),
    nn.Linear(2000, 500),
    nn.Sigmoid(),
    nn.Linear(500, 1)
)

def train_model():
    random.seed(0)
    train_data, train_cut, train_loader, X_train, y_train, X_test, y_test = load_data('../train_epoch=3/perlin_data.txt')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(model, train_loader, criterion, optimizer, epochs=50, device=device)
    torch.save(model.state_dict(), './feed_forward.pth')


    # EVALUATE MODEL
    y_pred = model.forward(X_test.to(device))
    loss = criterion(y_pred, y_test.to(device))

    print(loss.item())

    num_trainable = [get_num_trainable(data['densities']) for data in train_data][train_cut:]
    plt.scatter(num_trainable, y_test.detach().cpu().numpy(), label='Data points')
    plt.scatter(num_trainable, y_pred.detach().cpu().numpy(), label='Predictions')
    plt.text(
        1, 1.05, f'Loss: {loss.item():.7f}',
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=12, color="red"
    )
    plt.legend(loc='upper right')
    plt.xlabel('Num. Trainable Parameters')
    plt.ylabel('Test Loss')
    plt.savefig('feed_forward.png')


if __name__ == '__main__':
    train_model()


