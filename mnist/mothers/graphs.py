from mnist.util import get_num_trainable
import matplotlib.pyplot as plt
import numpy as np


def test_vs_pred(
        y_test, y_pred, loss,
        ylim=(None, None),
        xlim=(None, None),
        show=True,
        save=None
):
    plt.cla()
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    # Plot test vs. Pred
    plt.scatter(y_test, y_pred)
    plt.xlabel('Test Error', fontsize=20)
    plt.ylabel('Prediction', fontsize=20)

    # Plot line y=x
    points = np.append(y_test, y_pred)
    x = np.linspace(points.min(), points.max(), 100)
    plt.plot(x, x, linestyle='--', color='orange')

    plt.ylim(*ylim)
    plt.xlim(*xlim)

    plt.text(
        1, 1.06, f'Loss: {loss:.7f}',
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=18, color="red"
    )

    if show:
        plt.show()
    if save is not None:
        plt.savefig(
            save,
            bbox_inches='tight',
            pad_inches=0.05,
            dpi=300,
        )


def num_train_vs_test_graph(
        y_test, y_pred, jsons_test, loss,
        show=True,
        save=None
):
    plt.cla()
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    num_trainable = [get_num_trainable(data['density_map']) for data in jsons_test]
    plt.scatter(num_trainable, y_test, label='Data points')
    plt.scatter(num_trainable, y_pred, label='Predictions')
    plt.text(
        1, 1.06, f'Loss: {loss:.7f}',
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=18, color="red"
    )
    plt.legend(loc='upper right', fontsize=13)
    plt.xlabel('Num. Trainable Parameters', fontsize=20)
    plt.ylabel('Test Loss', fontsize=20)

    if show:
        plt.show()
    if save is not None:
        plt.savefig(
            save,
            bbox_inches='tight',
            pad_inches=0.05,
            dpi=300,
        )