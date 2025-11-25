import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import iqr

from cifar10.util import get_num_trainable


def generate_graphs(
    data_file='../data/variance.txt',
    graph_file='variance.png',
):
    with open(data_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    partitions = {500: [], 300: [], 200: [], 100: [], 50: []}
    for data in dataset:
        partitions[data['block_size']].append(data)

    # for block_size, dataset in partitions.items():
    #     plt.axis('off')
    #     plt.imshow(dataset[0]['density_map'], cmap='gray')
    #     plt.savefig(f'./sparsity_map_{block_size}.png', bbox_inches='tight', pad_inches=0)

    means = []
    stds = []
    labels = []
    dists = []

    for block_size, dataset in partitions.items():
        num_input = int(7500 / block_size)
        num_hidden = int(3000 / block_size)
        sizes = [block_size] * (num_input + num_hidden) + [10]

        trainable_parameters = [get_num_trainable((data['density_map']), sizes=sizes, num_input=num_input) for data in dataset]
        test_losses = [data['test_losses'][-1] for data in dataset]

        means.append(np.array(test_losses).mean())
        stds.append(np.array(test_losses).std())
        labels.append(block_size)
        dists.append(test_losses)

        plt.scatter(
            trainable_parameters, test_losses,
            label=f'{block_size}',
            alpha=1,
            s=40,
            zorder=2,
        )

    plt.ylim(1.4, 1.6)
    plt.xlabel('Num. Trainable Parameters')
    plt.ylabel('Test Loss')
    plt.legend(loc='upper right')
    plt.savefig(graph_file)

    plt.clf()
    x = np.arange(len(means))
    plt.errorbar(x, means, yerr=stds, fmt='o', capsize=5)
    plt.xticks(x, [f"{labels[i]}" for i in x])
    plt.ylabel("Value")
    plt.title("Mean ± Std of Distributions")
    plt.show()

    plt.clf()
    plt.boxplot(dists, vert=True, patch_artist=True)
    plt.xticks(range(1, len(dists) + 1), [f"{labels[i]}" for i in range(len(dists))])
    plt.ylabel("Test loss")
    plt.ylim(1.4, 1.6)
    plt.xlabel("Box size")
    plt.show()

    plt.clf()
    iqrs = [iqr(np.array(data)) for data in dists]
    plt.plot(labels, iqrs)
    plt.ylabel("Test loss IQR")
    plt.xlabel("Box size")
    plt.ylim(0)
    plt.gca().invert_xaxis()
    plt.show()
    print(iqrs)




if __name__ == '__main__':
    generate_graphs(
        data_file='../data/variance.txt',
        graph_file='./variance.png',
    )





