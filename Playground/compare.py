import torch
from torch import nn, optim
import torch.nn.functional as F
from Playground.training.create_models import masked_model
from Sequential2D import IterativeSequential2D
from Playground.training import train, load_mnist
import matplotlib.pyplot as plt
import numpy as np


# Load Data
data_folder = "../data"
train_loader, test_loader = load_mnist(data_folder, flat=True, size=(30, 30))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from Playground.util import num_trainable_parameters


# Sparse Model
sizes = [900, 400, 200, 100, 10]
sparse_model = masked_model(sizes=sizes, sparsity=0.4018)
print(f'Sparse Trainable: {num_trainable_parameters(sparse_model)}')

# Feed Forward Model
I = nn.Identity()
f1 = nn.Linear(in_features=900, out_features=400)
f2 = nn.Linear(in_features=400, out_features=200)
f3 = nn.Linear(in_features=200, out_features=100)
f4 = nn.Linear(in_features=100, out_features=10)

#          2500  500   200   100   10
blocks = [[I,    None, None, None, None],
          [f1,   None, None, None, None],
          [None, f2,   None, None, None],
          [None, None, f3,   None, None],
          [None, None, None, f4,   None]]

feed_forward_model = IterativeSequential2D(blocks, F.relu, 4)
print(f'Feed Forward Trainable: {num_trainable_parameters(feed_forward_model)}')

# Train Feed Forward
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(feed_forward_model.parameters(), lr=0.0001)

feed_forward_losses, _, _ = train(feed_forward_model, train_loader, criterion, optimizer, epochs=1, nth_batch=1, device=device)

# Train Sparse
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(sparse_model.parameters(), lr=0.0001)

sparse_losses, _, _ = train(sparse_model, train_loader, criterion, optimizer, epochs=1, nth_batch=1, device=device)


iterations = np.arange(len(feed_forward_losses))
plt.figure(figsize=(10, 5))
plt.plot(iterations, feed_forward_losses, label='Model 1', color='blue', linewidth=1)
plt.plot(iterations, sparse_losses, label='Model 2', color='orange', linewidth=1)

# Labels and title
plt.xlabel('Batch Num.')
plt.ylabel('Loss')
plt.title('Loss over Batches')
plt.legend()

# Save as PNG
plt.savefig('training_metrics.png', dpi=300)





