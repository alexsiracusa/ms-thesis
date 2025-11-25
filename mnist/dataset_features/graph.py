import matplotlib.pyplot as plt
from mnist.util import get_num_trainable
import numpy as np
import json


def display_datasets(dataset_name, noise, data_dir='../data'):

    data_file = f'{data_dir}/{dataset_name}/{noise}.txt'
    with open(data_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    features_file = f'{data_dir}/{dataset_name}/dataset_features.json'
    with open(features_file, 'r') as f:
        dataset_features = json.load(f)

    trainable_parameters = [get_num_trainable((data['density_map'])) for data in dataset]
    test_losses = ([data['test_loss'] for data in dataset])

    plt.scatter(
        trainable_parameters, test_losses,
        alpha=0.5,
        s=5,
    )

    print(get_num_trainable(np.full((11, 36), 0.5)))
    print(11 * 36 * 100 * 100 * 0.5)

    plt.scatter(
        x=np.array([0.1, 0.25, 0.6]) * (11 * 36 * 100 * 100),
        y=[dataset_features['nn_loss_0.1_test'], dataset_features['nn_loss_0.25_test'], dataset_features['nn_loss_0.5_test']],
        label='Dataset Features',
        s=50,
    )

    plt.xlabel('Num. Trainable Parameters')
    plt.ylabel('Test Loss')
    plt.ylim(0, dataset_features['loss_uniform'] + 0.05)
    plt.legend()
    plt.savefig('./fig.png')
    plt.show()


if __name__ == "__main__":
    display_datasets('mnist', 'sparse_random')