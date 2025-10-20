from cifar10.util import get_num_trainable
import matplotlib.pyplot as plt


def test_vs_pred(
        y_test, y_pred, loss,
        ylim=(None, None),
        xlim=(None, None),
        show=True,
        save=None
):
    plt.scatter(y_test, y_pred)
    plt.xlabel('Test Error')
    plt.ylabel('Prediction')

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


def num_train_vs_test_graph(
        y_test, y_pred, jsons_test, loss,
        show=True,
        save=None
):
    num_trainable = [get_num_trainable(data['density_map']) for data in jsons_test]
    plt.scatter(num_trainable, y_test, label='Data points')
    plt.scatter(num_trainable, y_pred, label='Predictions')
    plt.text(
        1, 1.05, f'Loss: {loss:.7f}',
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=12, color="red"
    )
    plt.legend(loc='upper right')
    plt.xlabel('Num. Trainable Parameters')
    plt.ylabel('Test Loss')

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)