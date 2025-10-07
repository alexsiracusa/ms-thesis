import matplotlib.pyplot as plt
import json

from cifar10.util import get_num_trainable


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

    for i, dataset in enumerate(datasets):
        trainable_parameters = [get_num_trainable((data['densities'])) for data in dataset]
        test_losses = [data['test_loss'] for data in dataset]

        # trainable_parameters, test_losses = zip(*[
        #     (n_param, loss) for n_param, loss in zip(trainable_parameters, test_losses)
        #     if n_param > 1e6
        # ])

        plt.scatter(
            trainable_parameters, test_losses,
            label=labels[i] if labels is not None else None,
            alpha=0.5,
            s=5,
        )

    plt.xlabel('Num. Trainable Parameters')
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



