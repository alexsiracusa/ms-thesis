import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import numpy as np
import json

from cifar10.util import get_num_trainable


def rectangular_identity(rows, cols):
    I = np.eye(rows)
    if cols < rows:
        return I[:, :cols]
    else:
        return np.hstack([I, np.zeros((rows, cols - rows))])


def sum_singular_value(densities):
    densities = np.array(densities)
    rows, cols = densities.shape
    pad_rows = cols - rows

    # square = np.vstack([np.zeros((pad_rows, cols), dtype=densities.dtype), densities])
    square = np.vstack([rectangular_identity(pad_rows, cols), densities])
    degrees = np.sum(square, axis=1)
    degrees = np.diag(degrees)

    L = degrees - square

    singular_values = np.linalg.svd(square, compute_uv=False)
    singular_values = np.sort(singular_values)

    # return np.linalg.norm(square, ord='fro')
    return np.sum(singular_values)


def heatmap(x, y, z):
    bins = 50

    # Compute average z in each bin
    stat, x_edges, y_edges, binnumber = binned_statistic_2d(
        x, y, z, statistic='mean', bins=bins
    )

    # Plot the heatmap
    plt.imshow(
        stat.T,  # transpose so x matches horizontal axis
        origin='lower',  # put (0,0) at bottom-left
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect='auto',
        cmap='viridis'
    )
    plt.colorbar(label='Average z')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Heatmap of average z per bin')
    plt.show()


def generate_graphs(
        data_files=['./train_epoch=3/train_data.txt', './train_epoch=3/perlin_data.txt'],
        labels=None,
        graph_file='graph.png',
):
    datasets = []
    for file in data_files:
        with open(file, 'r') as f:
            dataset = [json.loads(line) for line in f]
            datasets.append(dataset)

    x = []
    y = []
    z = []
    for i, dataset in enumerate(datasets):
        trainable_parameters = [get_num_trainable((data['densities'])) for data in dataset]
        singular_values = [sum_singular_value((data['densities'])) for data in dataset]
        test_losses = [data['test_loss'] for data in dataset]

        plt.scatter(
            singular_values, test_losses,
            label=labels[i] if labels is not None else None,
            alpha=0.5,
            s=5,
        )

        # heatmap(np.array(trainable_parameters), np.array(singular_values), np.array(test_losses))

        x += trainable_parameters
        y += singular_values
        z += test_losses

    # heatmap(np.array(x), np.array(y), np.array(z))


    plt.xlabel('Median Singular Values')
    plt.ylabel('Test Loss')
    plt.legend(loc='upper right')
    # plt.xscale('log')
    plt.savefig(graph_file)


if __name__ == '__main__':
    generate_graphs(
        data_files=['./train_epoch=3/perlin_data.txt', './train_epoch=3/train_data.txt'],
        labels=['Sparse perlin', 'Sparse random'],
        graph_file='graph.png',
    )



