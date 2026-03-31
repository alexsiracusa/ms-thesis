import matplotlib.pyplot as plt
import numpy as np
import json

from mnist.mothers import load_dataset
from mnist.util import get_num_trainable


def generate_graphs(
    dataset_name,
    graph_file='children.png',
):
    data_file = f'../data/{dataset_name}/sparse_perlin_generated.txt'

    background_file = f'../data/{dataset_name}/sparse_perlin.txt'

    with open(data_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    # with open(background_file, 'r') as f:
    #     background = [json.loads(line) for line in f]

    _, _, background = load_dataset(
        noise_types=['sparse_perlin'],
        feature_set=['density_map'],
        dataset_feature_set=[],
        include=[dataset_name],
        min_cut_off=0.33,
    )

    background_parameters = [get_num_trainable((data['density_map'])) for data in background]
    background_losses = [data['test_loss'] for data in background]

    trainable_parameters = [get_num_trainable((data['original_density_map'])) for data in dataset]
    test_losses = [data['original_test_loss'] for data in dataset]

    generated_parameters = [get_num_trainable(np.array(data['generated_density_map']).reshape(11, 36).clip(0,1)) for data in dataset]
    generated_losses = [data['generated_test_loss'] for data in dataset]

    plt.scatter(
        background_parameters, background_losses,
        label='Original Children',
        alpha=0.15,
        s=15,
        zorder=0,
    )

    plt.scatter(
        trainable_parameters, test_losses,
        label='Starting Points',
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

    # Feed forward baseline
    # plt.scatter(
    #     [9010935], [1.50110],
    #     # label='Baseline',
    #     alpha=1,
    #     s=40,
    #     zorder=3,
    #     c='gray',
    # )
    # plt.axhline(y=1.50110, color='gray', linestyle='--', label=f'Feed-Forward Baseline')

    # plt.xlim(left=0.5e6, right=None)
    # plt.ylim(top=2, bottom=None)
    plt.xlabel('Num. Trainable Parameters')
    plt.ylabel('Test Loss')
    plt.legend(loc='upper right')
    plt.savefig(graph_file)


if __name__ == '__main__':
    generate_graphs(
        dataset_name='blood_mnist',
        graph_file='./children.png',
    )



