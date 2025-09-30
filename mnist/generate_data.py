from mnist.util.sizes import *
from mnist.util import load_mnist, train_mnist, sparse_random
from Sequential2D.util import build_sequential2d
import json


def generate_data(
        data_folder='../data',
        output_file='./train_data.txt',
):
    train_loader, test_loader = load_mnist(
        data_folder,
        dataset='MNIST',
        flatten=True,
        batch_size=128
    )

    model_densities = []
    for _ in range(25):
        model_densities.append(sparse_random((num_blocks, num_blocks), 0.33))

    for densities in model_densities:
        for _ in range(25):
            model = build_sequential2d(
                sizes,
                type='linear',
                num_input_blocks=len(input_sizes),
                num_output_blocks=len(output_sizes),
                num_iterations=4,
                densities=densities.tolist(),
                weight_init='weighted',
            )

            train_loss, test_loss = train_mnist(
                model, train_loader, test_loader,
                epochs=1,
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
        output_file='./train_data.txt',
    )