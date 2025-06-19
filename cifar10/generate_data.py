import torch
import json

from train import train_cifar
from load_cifar import load_cifar


data_folder = "../data"
output_file = "./train_data.txt"

train_loader, test_loader = load_cifar(data_folder, batch_size=128)

input_sizes = [75] * 100
hidden_sizes = [50] * 44
output_sizes = [10]

num_blocks = len(input_sizes + hidden_sizes + output_sizes)
num_iterations = 4


for _ in range(300):
    densities = torch.empty((num_blocks, num_blocks)).uniform_(0, 1)

    train_loss, test_loss = train_cifar(
        input_sizes, hidden_sizes, output_sizes,
        num_iterations, densities,
        train_loader, test_loader
    )

    data = {
        "densities": densities[len(input_sizes):],
        "train_loss": train_loss,
        "test_loss": test_loss,
    }

    with open(output_file, 'a') as f:
        f.write(json.dumps(data) + '\n')

