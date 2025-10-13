import json
import matplotlib.pyplot as plt
from mnist.util import get_num_trainable

def visualize_data(
    data_files=['./data/mnist/sparse_random.txt'],
    labels=None,
    graph_file='graph.png',
):
    datasets = []
    for file in data_files:
        with open(file, 'r') as f:
            dataset = [json.loads(line) for line in f]
            datasets.append(dataset)

    for i, dataset in enumerate(datasets):
        # trainable_parameters = [get_num_trainable((data['density_map'])) for data in dataset]
        trainable_parameters = [data['average_density'] for data in dataset]
        test_losses = [data['test_loss'] for data in dataset]

        plt.scatter(
            trainable_parameters, test_losses,
            label=labels[i] if labels is not None else None,
            alpha=0.5,
            s=5,
        )

    plt.xlabel('Num. Trainable Parameters')
    plt.ylabel('Test Loss')
    plt.legend(loc='upper right')
    plt.ylim(0, None)
    plt.show()
    # plt.savefig(graph_file)


if __name__ == '__main__':
    visualize_data(
        data_files=['./data/mnist/sparse_random.txt', './data/mnist/sparse_perlin.txt'],
        labels=['Sparse Random', 'Sparse Perlin'],
        graph_file='graph.png',
    )

