import matplotlib.pyplot as plt
import numpy as np
import json

input_sizes = [75] * 100
hidden_sizes = [50] * 44
output_sizes = [10]
sizes = input_sizes + hidden_sizes + output_sizes

num_parameters = np.array([
    [row_size * col_size for col_size in sizes]
    for row_size in sizes
])


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
        trainable_parameters = [
            (np.array(data['densities']) * num_parameters[len(input_sizes):]).sum()
            for data in dataset
        ]
        test_losses = [data['test_loss'] for data in dataset]
        # trainable_parameters, test_losses = zip(*[(n_param, loss) for n_param, loss in zip(trainable_parameters, test_losses) if n_param < 0.5e7])

        plt.scatter(
            trainable_parameters, test_losses,
            label=labels[i] if labels is not None else None,
            alpha=0.5,
            s=5,
        )

    plt.xlabel('Num. Trainable Parameters')
    plt.ylabel('Test Loss')
    plt.savefig(graph_file)


if __name__ == '__main__':
    generate_graphs(
        data_files=['./train_epoch=3/perlin_data.txt', './train_epoch=3/train_data.txt'],
        labels=['sparse_perlin', 'sparse random'],
        graph_file='graph.png',
    )



