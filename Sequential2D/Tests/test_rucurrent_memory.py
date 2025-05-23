import numpy as np
import torch
import torch.nn.functional as F

from Sequential2D import FlatRecurrentSequential2D, RecurrentSequential2D, MaskedLinear


def build_sequential2d(sizes, num_input_blocks=1, num_iterations=1, flat=True, densities=None):
    blocks = np.empty((len(sizes), len(sizes)), dtype=object)

    for i in range(len(sizes)):
        for j in range(len(sizes)):
            if i < num_input_blocks:
                if i == j:
                    blocks[i, j] = torch.nn.Identity()
                else:
                    blocks[i, j] = None
            else:
                density = densities[i][j] if densities is not None else 1
                blocks[i, j] = MaskedLinear.sparse_random(sizes[j], sizes[i], percent=density)

    if flat:
        return FlatRecurrentSequential2D(
            blocks,
            sizes,
            num_iterations=num_iterations,
            activations=[F.relu] * (len(sizes) - 1) + [None]
        )
    else:
        return RecurrentSequential2D(
            blocks,
            num_iterations=num_iterations,
            activations=[F.relu] * (len(sizes) - 1) + [None]
        )


# sizes = [2500, 2000, 2000, 1000]
sizes = [2500, 2500, 2500]
num_iterations = 4
device = torch.device('cuda')


# Normal Model
# model = build_sequential2d(sizes, num_input_blocks=1, num_iterations=num_iterations, flat=False)
# data = [[torch.zeros((2, 2500), device=device, dtype=torch.float32)]] * 20
# model = model.to(device)
#
# for _ in range(1):
#     output = model.forward(data)
#     print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")


# Flat Model
model = build_sequential2d(sizes, num_input_blocks=1, num_iterations=num_iterations, flat=True)
data = torch.zeros((2, 20, 2500))

model = model.to(device)
data = data.to(device)

for _ in range(1):
    output = model.forward(data, batch_first=True)
    print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")


# Default RNN
# model = torch.nn.RNN(sizes[0], hidden_size=sum(sizes[1:]))
# data = torch.zeros((2, 20 * num_iterations, 2500))
#
# model = model.to(device)
# data = data.to(device)
#
# for _ in range(1):
#     output = model.forward(data)
#     print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

print(f"Peak Memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")




