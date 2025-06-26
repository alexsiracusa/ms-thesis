import json

from cifar10.util.train import train_cifar
from cifar10.util.load_cifar import load_cifar
from cifar10.util.random_densities import sparse_random, sparse_perlin
from util import input_sizes, hidden_sizes, output_sizes


def generate_data(
        data_folder='../data',
        output_file='./train_data.txt',
        density_fn=sparse_random,
):
    train_loader, test_loader = load_cifar(data_folder, batch_size=128, shuffle=False)

    num_blocks = len(input_sizes + hidden_sizes + output_sizes)
    num_iterations = 4

    for _ in range(999):
        densities = density_fn((num_blocks, num_blocks))

        train_loss, test_loss = train_cifar(
            input_sizes, hidden_sizes, output_sizes,
            num_iterations, densities,
            train_loader, test_loader,
            num_epochs=3,
        )

        data = {
            "densities": densities[len(input_sizes):].tolist(),
            "train_loss": train_loss,
            "test_loss": test_loss,
        }

        with open(output_file, 'a') as f:
            f.write(json.dumps(data) + '\n')

        print("")


if __name__ == '__main__':
    generate_data(
        data_folder='../data',
        output_file='train_epoch=3/train_data.txt',
        density_fn=sparse_random,
    )

