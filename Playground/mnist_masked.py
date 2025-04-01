import torch
from torch import nn, optim
import torch.nn.functional as F
from training import train, load_mnist
from Sequential2D import MaskedLinear, SparseAdam, IterativeSequential2D
from util import num_trainable_parameters
import numpy as np
import matplotlib.pyplot as plt
import copy



data_folder = "../data"
train_loader, test_loader = load_mnist(data_folder, flat=True)


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
                MaskedLinear.sparse_random(sizes[j], sizes[i], percent=0.5108),
            )

#            2500  500   200   100   10
# blocks = [[I,    None, None, None, None],
#           [f10,  f11,  f12,  f13,  f14 ],
#           [f20,  f21,  f22,  f23,  f24 ],
#           [f30,  f31,  f32,  f33,  f34 ],
#           [f40,  f41,  f42,  f43,  f44 ]]

model1 = IterativeSequential2D(blocks, 4, F.relu)
model2 = copy.deepcopy(model1)
print(f'Trainable: {num_trainable_parameters(model1)}')


# train
criterion = nn.CrossEntropyLoss()
adam = optim.Adam(model1.parameters(), lr=0.0001)
sparse_adam = SparseAdam(model2.parameters(), lr=0.0001)

adam_losses, _, _ = train(model1, train_loader, test_loader, criterion, adam, print_every_nth_batch=1)
sparse_adam_losses, _, _ = train(model2, train_loader, test_loader, criterion, sparse_adam, print_every_nth_batch=1)

iterations = np.arange(len(sparse_adam_losses))


plt.figure(figsize=(10, 5))
plt.plot(iterations, adam_losses, label='Loss (Adam)', color='blue')
plt.plot(iterations, sparse_adam_losses, label='Loss (SparseAdam)', color='red')

# Labels and title
plt.xlabel('Batch Num.')
plt.ylabel('Loss')
plt.title('Loss over Batches')
plt.legend()

# Save as PNG
plt.savefig('training_metrics.png', dpi=300)
plt.show()

