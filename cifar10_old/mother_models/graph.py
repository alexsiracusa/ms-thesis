import matplotlib.pyplot as plt
import numpy as np


def test_vs_pred(
        y_test, y_pred, loss,
        ylim=(None, None),
        xlim=(None, None),
        show=True,
        save=None
):
    # Plot test vs. Pred
    plt.scatter(y_test, y_pred)
    plt.xlabel('Test Error')
    plt.ylabel('Prediction')

    # Plot line y=x
    points = np.append(y_test, y_pred)
    x = np.linspace(points.min(), points.max(), 100)
    plt.plot(x, x, linestyle='--', color='orange')

    plt.ylim(*ylim)
    plt.xlim(*xlim)

    plt.text(
        1, 1.05, f'Loss: {loss:.7f}',
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=12, color="red"
    )

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)



from cifar10_old.mother_models.load_data import load_data
from cifar10_old.mother_models.feed_forward_nn import model
import torch

model.load_state_dict(torch.load("../mother_models/feed_forward.pth"))

for param in model.parameters():
    param.requires_grad = False


train_data, train_cut, train_loader, X_train, y_train, X_test, y_test = load_data('../train_epoch=3/perlin_data.txt')

y_pred = model(X_test)
test_vs_pred(y_test, y_pred, 0)


