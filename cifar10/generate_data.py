import json

from cifar10.util.train import train_cifar
from cifar10.util.load_cifar import load_cifar
from cifar10.util.random_densities import random_densities


data_folder = "../data"
output_file = "./train_data.txt"

train_loader, test_loader = load_cifar(data_folder, batch_size=128, shuffle=False)

input_sizes = [75] * 100
hidden_sizes = [50] * 44
output_sizes = [10]

num_blocks = len(input_sizes + hidden_sizes + output_sizes)
num_iterations = 4


for _ in range(999):
    densities = random_densities()

    train_loss, test_loss = train_cifar(
        input_sizes, hidden_sizes, output_sizes,
        num_iterations, densities,
        train_loader, test_loader
    )

    data = {
        "densities": densities[len(input_sizes):].tolist(),
        "train_loss": train_loss,
        "test_loss": test_loss,
    }

    with open(output_file, 'a') as f:
        f.write(json.dumps(data) + '\n')

    print("")

