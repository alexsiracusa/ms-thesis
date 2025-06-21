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
        data_file='./train_data.txt',
        graph_file='graph.png',
):
    with open(data_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    trainable_parameters = [
        (np.array(data['densities']) * num_parameters[len(input_sizes):]).sum()
        for data in dataset
    ]
    test_losses = [data['test_loss'] for data in dataset]
    # trainable_parameters, test_losses = zip(*[(n_param, loss) for n_param, loss in zip(trainable_parameters, test_losses) if n_param < 0.5e7])

    plt.clf()
    plt.scatter(trainable_parameters, test_losses)
    plt.xlabel('Num. Trainable Parameters')
    plt.ylabel('Test Loss')
    plt.savefig(graph_file)


if __name__ == '__main__':
    generate_graphs('./train_data.txt', 'graph.png')
    generate_graphs('./perlin_data.txt', 'perlin_graph.png')



