from util import random_boolean_tensor
from torch import nn
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

device = torch.device('cuda')
tensor_dim = 2000

sparsity_values = np.arange(0, 1.1, 0.1)
dense_times = []
sparse_times = []

linear = nn.Linear(tensor_dim, tensor_dim, bias=False)
full_tensor = linear.weight

for sparsity in sparsity_values:
    mask = random_boolean_tensor(tensor_dim, tensor_dim, int(tensor_dim**2 * sparsity))

    tensor = (full_tensor * mask)
    dense = tensor.to_dense().to(device)
    sparse = tensor.to_sparse_csr().to(device)
    bias = torch.normal(0, 1, size=(1, tensor_dim)).to(device)

    X = torch.normal(0, 1, size=(100, tensor_dim)).to(device)

    start = time.time() # START TIMER
    for _ in range(100):
        X = F.linear(X, dense, bias)
    dense_times.append(time.time() - start) # END TIMER

    X = torch.normal(0, 1, size=(100, tensor_dim)).to(device)

    start = time.time()  # START TIMER
    for _ in range(100):
        X = F.linear(X, sparse, bias)
    sparse_times.append(time.time() - start)  # END TIMER

print(sparse_times)
print(dense_times)

plt.figure(figsize=(10, 5))
plt.plot(sparsity_values, dense_times, label='Dense', color='red')
plt.plot(sparsity_values, sparse_times, label='Sparse', color='blue')

# Labels and title
plt.xlabel('Sparsity')
plt.ylabel('Time')
plt.title('Dot Product Time vs. Sparsity')
plt.legend()

# Save as PNG
plt.savefig('dot_times.png', dpi=300)
plt.show()
