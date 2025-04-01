import torch
import torch.nn.functional as F
from torch import nn, optim
from Sequential2D import IterativeSequential2D, SparseLinear, SparseAdam
from util import num_trainable_parameters
import numpy as np
from training import train, load_mnist


data_folder = "../data"
train_loader, test_loader = load_mnist(data_folder, flat=True)


# normal trainable: 1371810
# full trainable:   2685150

sizes = [2500, 500, 200, 100, 10]
blocks = np.empty((len(sizes), len(sizes)), dtype=object)

for i in range(len(sizes)):
    for j in range(len(sizes)):
        if i == 0 and j == 0:
            blocks[i, j] = torch.nn.Identity()
        elif i == 0:
            blocks[i, j] = None
        else:
            blocks[i, j] = nn.Sequential(
                SparseLinear.sparse_random(sizes[j], sizes[i], percent=0.5108),
            )

#            2500  500   200   100   10
# blocks = [[I,    None, None, None, None],
#           [f10,  f11,  f12,  f13,  f14 ],
#           [f20,  f21,  f22,  f23,  f24 ],
#           [f30,  f31,  f32,  f33,  f34 ],
#           [f40,  f41,  f42,  f43,  f44 ]]

model = IterativeSequential2D(blocks, 4, F.relu)
print(f'Trainable: {num_trainable_parameters(model)}')


# train
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
optimizer = SparseAdam(model.parameters(), lr=0.0001)

losses, forward_times, backward_times = train(model, train_loader, test_loader, criterion, optimizer, print_every_nth_batch=1)
