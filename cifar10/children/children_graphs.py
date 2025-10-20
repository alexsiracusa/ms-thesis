import matplotlib.pyplot as plt
import numpy as np
import json

from cifar10.util import get_num_trainable


def generate_graphs(
    data_file='../data/sparse_perlin_generated.txt',
    background_file='../data/sparse_perlin.txt',
    graph_file='children.png',
):
    with open(data_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    with open(background_file, 'r') as f:
        background = [json.loads(line) for line in f]

    background_parameters = [get_num_trainable((data['density_map'])) for data in background]
    background_losses = [data['test_losses'][-1] for data in background]

    trainable_parameters = [get_num_trainable((data['original_density_map'])) for data in dataset]
    test_losses = [data['original_test_losses'][-1] for data in dataset]

    generated_parameters = [get_num_trainable(np.array(data['generated_density_map']).reshape(11, 36).clip(0,1)) for data in dataset]
    generated_losses = [data['generated_test_losses'][-1] for data in dataset]

    plt.scatter(
        background_parameters, background_losses,
        label='Original',
        alpha=0.1,
        s=5,
        zorder=0,
    )

    plt.scatter(
        trainable_parameters, test_losses,
        label='Original',
        alpha=1,
        s=40,
        zorder=2,
    )

    plt.scatter(
        generated_parameters, generated_losses,
        label='Generated',
        alpha=1,
        s=40,
        zorder=2,
    )

    for x1, y1, x2, y2 in zip(trainable_parameters, test_losses, generated_parameters, generated_losses):
        plt.plot([x1, x2], [y1, y2], color='gray', linestyle='--', zorder=1, alpha=0.45)

    plt.xlabel('Num. Trainable Parameters')
    plt.ylabel('Test Loss')
    plt.legend(loc='upper right')
    plt.savefig(graph_file)


if __name__ == '__main__':
    generate_graphs(
        data_file='../data/sparse_perlin_generated.txt',
        background_file='../data/sparse_perlin.txt',
        graph_file='./children.png',
    )



