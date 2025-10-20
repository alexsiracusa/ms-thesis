import json
import numpy as np
import matplotlib.pyplot as plt
from cifar10.util import get_num_trainable
import cifar10_old.util as util


def graph_data(
    data_files=['./data/sparse_random.txt'],
    labels=None,
    graph_file='graph.png',
):
    datasets = []
    for file in data_files:
        with open(file, 'r') as f:
            dataset = [json.loads(line) for line in f]
            datasets.append(dataset)

    for i, dataset in enumerate(datasets[:2]):
        trainable_parameters = [get_num_trainable((data['density_map'])) for data in dataset]
        # trainable_parameters = [data['average_density'] for data in dataset]
        test_losses = ([data['test_losses'][-1] for data in dataset])

        plt.scatter(
            trainable_parameters, test_losses,
            label=labels[i] if labels is not None else None,
            alpha=0.5,
            s=5,
        )

    for i, dataset in enumerate(datasets[2:]):
        trainable_parameters = [util.get_num_trainable((data['densities'])) for data in dataset]
        # trainable_parameters = [data['average_density'] for data in dataset]
        test_losses = ([data['test_loss'] for data in dataset])

        plt.scatter(
            trainable_parameters, test_losses,
            label=labels[i+2] if labels is not None else None,
            alpha=0.5,
            s=5,
        )

    plt.xlabel('Num. Trainable Parameters')
    plt.ylabel('Test Loss')
    plt.legend(loc='upper right')

    if graph_file is not None:
        plt.savefig(graph_file)

    plt.show()


def show_noises(data_file):
    with open(data_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    arr = np.random.choice(dataset, size=6)

    fig, axes = plt.subplots(3, 2, figsize=(5, 4), facecolor="black")

    for data, ax in zip(arr, axes.flat):
        density_map = data['density_map']
        test_loss = data['test_losses'][-1]

        ax.imshow(density_map, cmap='gray')
        ax.text(
            0, -3, f"{test_loss:.5f}",
            color='red', fontsize=8, weight='bold', ha='left', va='top'
        )
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    graph_data(
        data_files=[
            './data/sparse_random.txt',
            './data/sparse_perlin.txt',
            # '../cifar10_old/train_epoch=3/train_data.txt',
            # '../cifar10_old/train_epoch=3/perlin_data.txt',
        ],
        labels=[
            'Sparse Random',
            'Sparse Perlin',
            'Random Old',
            'Perlin Old',
        ],
        graph_file='graph.png',
    )

    # show_noises(f'./data/sparse_random.txt')

