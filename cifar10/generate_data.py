import torch
import json
import random

from train import train_cifar
from load_cifar import load_cifar


data_folder = "../data"
output_file = "./train_data.txt"

train_loader, test_loader = load_cifar(data_folder, batch_size=128, shuffle=False)

input_sizes = [75] * 100
hidden_sizes = [50] * 44
output_sizes = [10]

num_blocks = len(input_sizes + hidden_sizes + output_sizes)
num_iterations = 4


def random_densities():
    shape = (num_blocks, num_blocks)

    p_random = 0.25 * random.random() + 0.01
    print(f"p_random: {p_random}")

    random_tensor = torch.rand(shape)
    mask = torch.rand(shape) < p_random

    base_tensor = torch.zeros(shape)
    base_tensor[mask] = random_tensor[mask]

    return base_tensor


for _ in range(300):
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

