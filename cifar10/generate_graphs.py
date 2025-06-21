import matplotlib.pyplot as plt
import numpy as np
import json


def generate_graphs(
        data_file='./train_data.txt',
        graph_file='graph.png',
):
    with open(data_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    sum_densities = [sum(np.array(data['densities']).flatten()) for data in dataset]
    test_losses = [data['test_loss'] for data in dataset]

    plt.clf()
    plt.scatter(sum_densities, test_losses)
    plt.savefig(graph_file)


if __name__ == '__main__':
    generate_graphs('./train_data.txt', 'graph.png')
    generate_graphs('./perlin_data.txt', 'perlin_graph.png')



