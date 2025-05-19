from torch import nn, optim
from Sequential2D import IterativeSequential2D
from Playground.util import num_trainable_parameters
from Playground.training import load_mnist, train
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


data_folder = "../data"
train_loader, test_loader = load_mnist(data_folder, flat=True)


# define model
I = nn.Identity()
f1 = nn.Linear(in_features=2500, out_features=500)
f2 = nn.Linear(in_features=500, out_features=200)
f3 = nn.Linear(in_features=200, out_features=100)
f4 = nn.Linear(in_features=100, out_features=10)

#          2500  500   200   100   10
blocks = [[I,    None, None, None, None],
          [f1,   None, None, None, None],
          [None, f2,   None, None, None],
          [None, None, f3,   None, None],
          [None, None, None, f4,   None]]

model1 = IterativeSequential2D(blocks, 4, F.relu)
print(f'Trainable: {num_trainable_parameters(model1)}')


# train
criterion = nn.CrossEntropyLoss()
optim = optim.Adam(model1.parameters(), lr=0.0001)

losses, _, _ = train(model1, train_loader, criterion, optim, nth_batch=1)
iterations = np.arange(len(losses))


plt.figure(figsize=(10, 5))
plt.plot(iterations, losses, label='Loss', color='blue')

# Labels and title
plt.xlabel('Batch Num.')
plt.ylabel('Loss')
plt.title('Loss over Batches')
plt.legend()

# Save as PNG
plt.savefig('training_metrics.png', dpi=300)
plt.show()
